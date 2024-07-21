import os
import logging
import tempfile
from typing import Tuple, List
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import torch
import librosa
import numpy as np
from team_code.const import TARGET_SAMPLING_FREQUENCY, PUNCTUATION
from team_code.model import MultimodalModel
from team_code.text import find_long_text_similar_to_short_text, ordinal_to_str, cardinal_to_str
from config import startup_config, runtime_config


audio_processing_logger = logging.getLogger("audio processing")


def transform_to_wavpcm(src_fname: str, dst_fname: str) -> None:
    found_idx = src_fname.rfind(".")
    if found_idx < 0:
        err_msg = (
            f'The extension of the file "{src_fname}" is unknown. '
            f"So, I cannot determine a format of this sound file."
        )
        raise ValueError(err_msg)
    if not os.path.isfile(src_fname):
        err_msg = f'The file "{src_fname}" does not exist!'
        raise IOError(err_msg)
    source_audio_extension = src_fname[(found_idx + 1) :]
    try:
        audio = AudioSegment.from_file(src_fname, format=source_audio_extension)
    except CouldntDecodeError as e1:
        audio = None
        additional_err_msg = str(e1)
    except BaseException as e2:
        audio = None
        additional_err_msg = str(e2)
    else:
        additional_err_msg = ""
    if audio is None:
        err_msg = f'The file "{src_fname}" cannot be opened.'
        if additional_err_msg != "":
            err_msg += f" {additional_err_msg}"
        raise IOError(err_msg)
    if audio.channels != 1:
        audio.set_channels(1)
    if audio.frame_rate != TARGET_SAMPLING_FREQUENCY:
        audio.set_frame_rate(TARGET_SAMPLING_FREQUENCY)
    if audio.frame_width != 2:
        audio.set_sample_width(2)
    target_parameters = [
        "-ac",
        "1",
        "-ar",
        f"{TARGET_SAMPLING_FREQUENCY}",
        "-acodec",
        "pcm_s16le",
    ]
    audio.export(dst_fname, format="wav", parameters=target_parameters)


def load_sound(audio_fname: str) -> Tuple[np.ndarray, str]:
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".wav") as fp:
        tmp_wav_name = fp.name
    try:
        transform_to_wavpcm(audio_fname, tmp_wav_name)
    except BaseException as ex:
        err_msg = str(ex)
        audio_processing_logger.error(err_msg)
        raise
    audio_processing_logger.info(
        f'The sound "{audio_fname}" is converted to the "{tmp_wav_name}".'
    )
    try:
        input_sound, _ = librosa.load(
            path=tmp_wav_name, sr=TARGET_SAMPLING_FREQUENCY, dtype=np.float32
        )
    except BaseException as ex:
        err_msg = str(ex)
        audio_processing_logger.error(err_msg)
        raise
    audio_processing_logger.info(f'The sound is "{tmp_wav_name}" is loaded.')
    return input_sound, tmp_wav_name


def generate_audio_caption(
    audio_fname: str, model: MultimodalModel
) -> Tuple[str, bool]:
    sound, tmp_sound_fname = load_sound(audio_fname)
    try:
        if startup_config.DEVICE.type == "cpu":
            inputs = model.audio[0](
                [sound.tolist()],
                sampling_rate=TARGET_SAMPLING_FREQUENCY,
                return_tensors="pt",
            ).to(startup_config.DEVICE)
        else:
            inputs = model.audio[0](
                [sound.tolist()],
                sampling_rate=TARGET_SAMPLING_FREQUENCY,
                return_tensors="pt",
            ).to(startup_config.DEVICE, torch.float16)

        with torch.no_grad():
            logits = model.audio[1](**inputs).logits
        del inputs
        predicted_class_ids = int(torch.argmax(logits, dim=-1).item())
        del logits
        predicted_label = model.audio[1].config.id2label[predicted_class_ids]
        audio_caption = " ".join(" ".join(predicted_label.split("_")).split())
        if (audio_caption.lower().find("speech") >= 0) and (
            audio_caption.lower().find("speech noise") < 0
        ):
            audio_caption = model.speech(sound)["text"]
            is_speech = True
        else:
            is_speech = False
        del sound
    finally:
        if (len(tmp_sound_fname) > 0) and os.path.isfile(tmp_sound_fname):
            os.remove(tmp_sound_fname)
    return audio_caption, is_speech


def find_text_by_audio(audio_fname: str, model: MultimodalModel) -> Tuple[str, bool]:
    if not os.path.isfile(audio_fname):
        err_msg = f'The image "{audio_fname}" does not exist!'
        audio_processing_logger.error(err_msg)
        raise ValueError(err_msg)
    audio_caption, is_speech = generate_audio_caption(audio_fname, model)
    audio_caption = audio_caption.strip()
    if audio_caption[-1] not in {".", "!", "?"}:
        audio_caption = audio_caption + "."
    if not audio_caption[0].isupper():
        audio_caption = audio_caption[0].upper() + audio_caption[1:]
    if is_speech:
        return audio_caption, True

    if runtime_config.use_one_peace and startup_config.load_one_peace:
        src_audios, audio_padding_masks = model.one_peace.process_audio([audio_fname])
        with torch.no_grad():
            audio_features = model.one_peace.extract_audio_features(
                src_audios, audio_padding_masks
            )
        del src_audios, audio_padding_masks
        audio_vector = model.pca.transform(
            audio_features.cpu().type(torch.FloatTensor).numpy()[0:1]
        )[0]
        del audio_features
        found_indices = model.annoy_index.get_nns_by_vector(
            audio_vector,
            n=runtime_config.max_wiki_paragraphs,
            search_k=runtime_config.annoy_search_k,
        )
        del audio_vector
        found_texts = [model.texts[idx] for idx in found_indices]
        del found_indices
        if len(found_texts) > 1:
            if len(found_texts) == 1:
                long_text = found_texts[0]
            else:
                long_text = find_long_text_similar_to_short_text(
                    audio_caption, found_texts, model
                )
        else:
            long_text = ""

    result = ". ".join(filter(bool, (audio_caption, long_text)))
    # if len(found_texts) > 1:
    #     result = audio_caption + ' ' + find_long_text_similar_to_short_text(audio_caption, found_texts, model)
    # else:
    #     result = audio_caption + ' ' + found_texts[0]
    return " ".join(result.split()), False


def generate_prompt_for_audio(audio_description: List[Tuple[str, bool]]) -> str:
    if len(audio_description) < 1:
        return ''
    if len(audio_description) == 1:
        if audio_description[0][1]:
            prompt = (f'I have just heard a sound that probably contains the '
                      f'following speech. {audio_description[0][0]}')
        else:
            prompt = (f'I have just heard a sound that probably corresponds to '
                      f'the following text description. {audio_description[0][0]}')
        if prompt[-1] not in PUNCTUATION:
            prompt += '.'
        elif prompt[-1] not in {'.', '?', '!'}:
            prompt = prompt[:-1] + '.'
    else:
        prompt = f'I have just heard {cardinal_to_str(len(audio_description))} sounds.'
        counter = 1
        for it in audio_description:
            if it[1]:
                prompt += (f' The {ordinal_to_str(counter)} sound probably contains the '
                           f'following speech. {it[0]}')
            else:
                prompt += (f' The {ordinal_to_str(counter)} sound probably corresponds to '
                           f'the following text description. {it[0]}')
            if prompt[-1] not in PUNCTUATION:
                prompt += '.'
            elif prompt[-1] not in {'.', '?', '!'}:
                prompt = prompt[:-1] + '.'
            counter += 1
    return prompt + ' Please imagine that you have just heard the same.'
