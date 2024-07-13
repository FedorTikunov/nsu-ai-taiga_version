import fileinput
import pickle
from collections import namedtuple
import gc
import logging
import os
import sys
import tempfile
from typing import Dict, List, Tuple
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
import numpy as np
from annoy import AnnoyIndex
import librosa
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_distances
from sklearn.pipeline import Pipeline
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from PIL import Image
import config
from torchvision import transforms
from team_code.ONE_PEACE.one_peace.models import from_pretrained

DEVICE = torch.device("cuda:0")
# DEVICE = torch.device("cpu")
TARGET_SAMPLING_FREQUENCY = 16_000

MultimodalModel = namedtuple(
    'MultimodalModel',
    'image audio speech sbert one_peace pca annoy_index texts ocr llm translate_ruen translate_enru'
)
conversation_logger = logging.getLogger(__name__)
PUNCTUATION = {'.', '?', '!', ':', '-', ';'}
CARDINAL_UNITS = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
                  '8': 'eight', '9': 'nine'}
CARDINAL_TEENS = {'10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen', '15': 'fifteen',
                  '16': 'sixteen', '17': 'seventeen', '18': 'eighteen', '19': 'nineteen'}
CARDINAL_TENS = {'2': 'twenty', '3': 'thirty', '4': 'forty', '5': 'fifty', '6': 'sixty', '7': 'seventy',
                 '8': 'eighty', '9': 'ninety'}
ORDINAL_UNITS = {'1': 'first', '2': 'second', '3': 'third', '4': 'fourth', '5': 'fifth', '6': 'sixth', '7': 'seventh',
                 '8': 'eighth', '9': 'ninth'}
ORDINAL_TEENS = {'10': 'tenth', '11': 'eleventh', '12': 'twelfth', '13': 'thirteenth', '14': 'fourteenth',
                 '15': 'fifteenth', '16': 'sixteenth', '17': 'seventeenth', '18': 'eighteenth', '19': 'nineteenth'}
ORDINAL_TENS = {'20': 'twentieth', '30': 'thirtieth', '40': 'fortieth', '50': 'fiftieth', '60': 'sixtieth',
                '70': 'seventieth', '80': 'eightieth', '90': 'ninetieth'}


def generate_image_caption(image_fname: str, model: MultimodalModel) -> str:
    raw_image = Image.open(image_fname).convert('RGB')
    if DEVICE.type == "cpu":
        inputs = model.image[0](raw_image, return_tensors="pt").to(DEVICE)
    else:    
        inputs = model.image[0](raw_image, return_tensors="pt").to(DEVICE, torch.float16)
    out = model.image[1].generate(**inputs)
    output = model.image[0].decode(out[0], skip_special_tokens=True)
    return output


def transform_to_wavpcm(src_fname: str, dst_fname: str) -> None:
    found_idx = src_fname.rfind('.')
    if found_idx < 0:
        err_msg = f'The extension of the file "{src_fname}" is unknown. ' \
                  f'So, I cannot determine a format of this sound file.'
        raise ValueError(err_msg)
    if not os.path.isfile(src_fname):
        err_msg = f'The file "{src_fname}" does not exist!'
        raise IOError(err_msg)
    source_audio_extension = src_fname[(found_idx + 1):]
    try:
        audio = AudioSegment.from_file(src_fname, format=source_audio_extension)
    except CouldntDecodeError as e1:
        audio = None
        additional_err_msg = str(e1)
    except BaseException as e2:
        audio = None
        additional_err_msg = str(e2)
    else:
        additional_err_msg = ''
    if audio is None:
        err_msg = f'The file "{src_fname}" cannot be opened.'
        if additional_err_msg != '':
            err_msg += f' {additional_err_msg}'
        raise IOError(err_msg)
    if audio.channels != 1:
        audio.set_channels(1)
    if audio.frame_rate != TARGET_SAMPLING_FREQUENCY:
        audio.set_frame_rate(TARGET_SAMPLING_FREQUENCY)
    if audio.frame_width != 2:
        audio.set_sample_width(2)
    target_parameters = ['-ac', '1', '-ar', f'{TARGET_SAMPLING_FREQUENCY}', '-acodec', 'pcm_s16le']
    audio.export(dst_fname, format='wav', parameters=target_parameters)


def load_sound(audio_fname: str) -> Tuple[np.ndarray, str]:
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.wav') as fp:
        tmp_wav_name = fp.name
    try:
        transform_to_wavpcm(audio_fname, tmp_wav_name)
    except BaseException as ex:
        err_msg = str(ex)
        conversation_logger.error(err_msg)
        raise
    conversation_logger.info(f'The sound "{audio_fname}" is converted to the "{tmp_wav_name}".')
    try:
        input_sound, _ = librosa.load(path=tmp_wav_name, sr=TARGET_SAMPLING_FREQUENCY, dtype=np.float32)
    except BaseException as ex:
        err_msg = str(ex)
        conversation_logger.error(err_msg)
        raise
    conversation_logger.info(f'The sound is "{tmp_wav_name}" is loaded.')
    return input_sound, tmp_wav_name


def generate_audio_caption(audio_fname: str, model: MultimodalModel) -> Tuple[str, bool]:
    sound, tmp_sound_fname = load_sound(audio_fname)
    try:
        if DEVICE.type == "cpu":
            inputs = model.audio[0](
                [sound.tolist()],
                sampling_rate=TARGET_SAMPLING_FREQUENCY, return_tensors="pt").to(DEVICE)
        else:
            inputs = model.audio[0](
                [sound.tolist()],
                sampling_rate=TARGET_SAMPLING_FREQUENCY, return_tensors="pt").to(DEVICE, torch.float16)
            
        with torch.no_grad():
            logits = model.audio[1](**inputs).logits
        del inputs
        predicted_class_ids = int(torch.argmax(logits, dim=-1).item())
        del logits
        predicted_label = model.audio[1].config.id2label[predicted_class_ids]
        audio_caption = ' '.join(' '.join(predicted_label.split('_')).split())
        if (audio_caption.lower().find('speech') >= 0) and (audio_caption.lower().find('speech noise') < 0):
            audio_caption = model.speech(sound)['text']
            is_speech = True
        else:
            is_speech = False
        del sound
    finally:
        if (len(tmp_sound_fname) > 0) and os.path.isfile(tmp_sound_fname):
            os.remove(tmp_sound_fname)
    return audio_caption, is_speech


def find_long_text_similar_to_short_text(short_text: str, long_texts: List[str], model: MultimodalModel) -> str:
    indices_of_long_texts = []
    all_sentences = []
    for idx, txt in enumerate(long_texts):
        sentences_in_text = sent_tokenize(txt)
        for cur_sent in sentences_in_text:
            all_sentences.append(cur_sent)
            indices_of_long_texts.append(idx)
        del sentences_in_text
    sentence_embeddings_of_long_texts = model.sbert.encode(all_sentences)
    del all_sentences
    sentence_embeddings_of_short_text = model.sbert.encode([short_text])
    distances = cosine_distances(X=sentence_embeddings_of_short_text, Y=sentence_embeddings_of_long_texts)[0].tolist()
    conversation_logger.info(f"indices22: {indices_of_long_texts}")
    conversation_logger.info(f"dists22: {distances}")
    del sentence_embeddings_of_short_text, sentence_embeddings_of_long_texts
    sentences_with_distances = sorted(
        list(zip(indices_of_long_texts, distances)),
        key=lambda it: (it[1], it[0])
    )
    del distances, indices_of_long_texts
    found_idx = sentences_with_distances[0][0]
    return long_texts[found_idx]

def extract_text_with_trocr(image_fname: str, model: MultimodalModel) -> str:
    # Load the image
    image = Image.open(image_fname)

    image = image.convert("RGB")
    
    
    # Process the image
    pixel_values = model.ocr[0](image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(DEVICE)
    generated_ids = model.ocr[1].generate(pixel_values)

    # Generate text
    text = model.ocr[0].batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text


def find_text_by_image(image_fname: str, model: MultimodalModel, top_n: int = 10, search_k: int = -1) -> str:
    if not os.path.isfile(image_fname):
        err_msg = f'The image "{image_fname}" does not exist!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    image_caption = generate_image_caption(image_fname, model)

    if config.use_ocr:
        trocr_text = extract_text_with_trocr(image_fname, model)
        trocr_text = f'Image has such text: "{trocr_text}"'
    else:
        trocr_text = ''
    
    if config.use_one_peace:
        src_images = model.one_peace.process_image([image_fname])
        with torch.no_grad():
            image_features = model.one_peace.extract_image_features(src_images)
        del src_images
        image_vector = model.pca.transform(image_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
        del image_features
        found_indices, dists = model.annoy_index.get_nns_by_vector(image_vector, n=top_n, search_k=search_k, include_distances=True)
        conversation_logger.info(f"Annoy indices: {found_indices}")
        conversation_logger.info(f"Annoy dists: {dists}")
        del image_vector
        found_texts = [model.texts[idx] for idx in found_indices]
        del found_indices
        long_text = find_long_text_similar_to_short_text(image_caption, found_texts, model)
        if len(found_texts) > 1:
            if len(found_texts) == 1:
                long_text = found_texts[0]
            else:
                long_text = find_long_text_similar_to_short_text(image_caption, found_texts, model)
        else:
            long_text = ''
    else:
        long_text = ''

    if config.use_blit:
        result = ". ".join(filter(bool, (image_caption, long_text, trocr_text)))
    else:
        result = ". ".join(filter(bool, (long_text, trocr_text)))

    # if len(found_texts) > 1:
    #     result = image_caption + ' ' + find_long_text_similar_to_short_text(image_caption, found_texts, model)
    # else:
    #     result = image_caption + ' ' + found_texts[0]

    # if len(trocr_text) > 0:
    #     result = result + ' Image has such text: ' + '"' + trocr_text + '"'
    
    return ' '.join(result.split())


def find_text_by_audio(audio_fname: str, model: MultimodalModel, top_n: int = 100,
                       search_k: int = -1) -> Tuple[str, bool]:
    if not os.path.isfile(audio_fname):
        err_msg = f'The image "{audio_fname}" does not exist!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    audio_caption, is_speech = generate_audio_caption(audio_fname, model)
    audio_caption = audio_caption.strip()
    if audio_caption[-1] not in {'.', '!', '?'}:
        audio_caption = audio_caption + '.'
    if not audio_caption[0].isupper():
        audio_caption = audio_caption[0].upper() + audio_caption[1:]
    if is_speech:
        return audio_caption, True
    
    if config.use_one_peace:
        src_audios, audio_padding_masks = model.one_peace.process_audio([audio_fname])
        with torch.no_grad():
            audio_features = model.one_peace.extract_audio_features(src_audios, audio_padding_masks)
        del src_audios, audio_padding_masks
        audio_vector = model.pca.transform(audio_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
        del audio_features
        found_indices = model.annoy_index.get_nns_by_vector(audio_vector, n=top_n, search_k=search_k)
        del audio_vector
        found_texts = [model.texts[idx] for idx in found_indices]
        del found_indices
        if len(found_texts) > 1:
            if len(found_texts) == 1:
                long_text = found_texts[0]
            else:
                long_text = find_long_text_similar_to_short_text(audio_caption, found_texts, model)
        else:
            long_text = ''
    
    result = ". ".join(filter(bool, (audio_caption, long_text)))
    # if len(found_texts) > 1:
    #     result = audio_caption + ' ' + find_long_text_similar_to_short_text(audio_caption, found_texts, model)
    # else:
    #     result = audio_caption + ' ' + found_texts[0]
    return ' '.join(result.split()), False
    
def process_image(image_fname: str) -> torch.Tensor:
    if not os.path.isfile(image_fname):
        return None  # Return None if the image file does not exist
    
    # Open the image file
    with Image.open(image_fname) as img:
        # Define the transformations: resize and tensor conversion
        transform = transforms.Compose([
            transforms.Resize((672, 672)),  # Resize to the size expected by LLaVA-NeXT
            transforms.ToTensor(),  # Convert the PIL Image to a tensor
        ])
        
        # Apply the transformations
        image_tensor = transform(img)
        
    return image_tensor


def load_images(image_file_list: List[str]):
    
    if not image_file_list:
        return None
    return [np.array(Image.open(file).convert("RGB")) for file in image_file_list]

def tokenize_prompt(prompt: str, image_file_list: List[str], tokenizer: AutoTokenizer, add_eos_token: bool = True,
                    add_labels: bool=True) -> Dict[str, Tuple[List[int], List[torch.Tensor]]]:
    result = tokenizer(prompt, padding=False, return_tensors=None)
    if (result['input_ids'][-1] != tokenizer.eos_token_id) and add_eos_token:
        result['input_ids'].append(tokenizer.eos_token_id)
        result['attention_mask'].append(1)
    if add_labels:
        result['labels'] = result['input_ids'].copy()
    result['images'] = [process_image(image_fname) for image_fname in image_file_list if process_image(image_fname) is not None]
    return result

def generate_answer_based_on_prompt(prompt: str, image_file_list: List[str], model: LlavaNextForConditionalGeneration, processor: LlavaNextProcessor) -> str:
    '''
    tokenized_text = tokenize_prompt(
        prompt,
        image_file_list,
        tokenizer,
        add_labels=False
    )
    input_ids = [torch.tensor(data=tokenized_text['input_ids'], dtype=torch.long)]
    attention_mask = [torch.tensor(data=tokenized_text['attention_mask'], dtype=torch.long)]
    images = tokenized_text['images'] if tokenized_text['images'] else None
    del tokenized_text
    '''
    # TEMP!!! DO NOT USE Image.open. 

    images = load_images(image_file_list)
    
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(DEVICE)
    '''
    batched_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True, padding_value=0  # <unk> idx
    ).to(DEVICE)[:,:-1]
    batched_attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask,
        batch_first=True, padding_value=0
    ).to(DEVICE)[:,:-1]
    '''
    
    '''
    generated_ids = model.generate(
        input_ids=batched_input_ids, attention_mask=batched_attention_mask, images=images,
        max_new_tokens=1000, do_sample=True
    )
    '''
    
    generated_ids = model.generate(**inputs, max_new_tokens=1000, do_sample=True)

    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    input_prompt = processor.decode(inputs["input_ids"][0], skip_special_tokens=True)
    
    #del batched_input_ids, batched_attention_mask
    
    #predicted_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    #input_prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    
    #del input_ids, attention_mask, generated_ids
    
    if len(predicted_text) < len(input_prompt):
        err_msg = (f'The predicted answer "{predicted_text}" does not correct, '
                   f'because it does not start with the prompt "{input_prompt}".')
        raise ValueError(err_msg)
    if not predicted_text.startswith(input_prompt):
        err_msg = (f'The predicted answer "{predicted_text}" does not correct, '
                   f'because it does not start with the prompt "{input_prompt}".')
        raise ValueError(err_msg)

    return ' '.join(predicted_text[len(input_prompt):].split()).strip()


def generate_logits_based_on_prompt(prompt: str, image_file_list: List[str], model: LlavaNextForConditionalGeneration, processor: LlavaNextProcessor) -> torch.Tensor:
    '''
    tokenized_text = tokenize_prompt(
        prompt,
        image_file_list,
        tokenizer,
        add_labels=False
    )
    input_ids = [torch.tensor(data=tokenized_text['input_ids'], dtype=torch.long)]
    attention_mask = [torch.tensor(data=tokenized_text['attention_mask'], dtype=torch.long)]
    images = tokenized_text['images']
    del tokenized_text
    
    batched_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True, padding_value=0  # <unk> idx
    ).to(DEVICE)
    batched_attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask,
        batch_first=True, padding_value=0
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            images=images,
            return_dict=True
        ).logits
    '''

    images = load_images(image_file_list)
    
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs, return_dict=True).logits

    return logits.cpu().type(torch.FloatTensor)



def cardinal_to_str(value: int) -> str:
    if value >= 100:
        err_msg = f'{value} is too large number of objects (images or audios)!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    value_str = f'{value}'
    if len(value_str) == 1:
        return CARDINAL_UNITS[value_str]
    if value_str in CARDINAL_TEENS:
        return CARDINAL_TEENS[value_str]
    if value_str[-1] == '0':
        return CARDINAL_TENS[value_str[0]]
    return CARDINAL_TENS[value_str[0]] + ' ' + CARDINAL_UNITS[value_str[-1]]


def ordinal_to_str(value: int) -> str:
    if value >= 100:
        err_msg = f'{value} is too large number of objects (images or audios)!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    value_str = f'{value}'
    if len(value_str) == 1:
        return ORDINAL_UNITS[value_str]
    if value_str in ORDINAL_TEENS:
        return ORDINAL_TEENS[value_str]
    if value_str in ORDINAL_TENS:
        return ORDINAL_TENS[value_str]
    return CARDINAL_TENS[value_str[0]] + ' ' + ORDINAL_UNITS[value_str[-1]]


def generate_prompt_for_image(image_description: List[str]) -> str:
    if len(image_description) < 1:
        return ''
    if len(image_description) == 1:
        prompt = (f'I have just looked at an <image> that probably corresponds to '
                  f'the following text description. {image_description[0]}')
        if prompt[-1] not in PUNCTUATION:
            prompt += '.'
        elif prompt[-1] not in {'.', '?', '!'}:
            prompt = prompt[:-1] + '.'
    else:
        prompt = f'I have just looked at {cardinal_to_str(len(image_description))} images.'
        counter = 1
        for it in image_description:
            prompt += (f' The {ordinal_to_str(counter)} <image> probably corresponds to '
                       f'the following text description. {it}')
            if prompt[-1] not in PUNCTUATION:
                prompt += '.'
            elif prompt[-1] not in {'.', '?', '!'}:
                prompt = prompt[:-1] + '.'
            counter += 1
    return prompt + ' Please imagine that you have just seen the same.'


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


def parse_query(cur_query_list: List[Dict[str, str]]) -> Tuple[List[str], List[str], List[str]]:
    possible_content = {'text', 'image', 'audio'}
    text_list = []
    image_file_list = []
    audio_file_list = []
    for i, q in enumerate(cur_query_list):
        if 'type' not in q:
            err_msg = f'The query {i} = {q} is wrong, because it does not contain the "type" field!'
            conversation_logger.error(err_msg)
            raise ValueError(err_msg)
        if 'content' not in q:
            err_msg = f'The query {i} = {q} is wrong, because it does not contain the "content" field!'
            conversation_logger.error(err_msg)
            raise ValueError(err_msg)
        if q['type'] not in possible_content:
            err_msg = (f'The query {i} = {q} is wrong, because the content type "{q["type"]}" is unknown! '
                       f'Expected text, image or audio.')
            conversation_logger.error(err_msg)
            raise ValueError(err_msg)
        if q['type'] == 'text':
            new_text = q['content'].strip()
            if len(new_text) == 0:
                err_msg = f'The query {i} = {q} is wrong, because the content is empty.'
                conversation_logger.error(err_msg)
                raise ValueError(err_msg)
            new_text = ' '.join(new_text.split())
            text_list.append(new_text)
        elif q['type'] == 'image':
            new_fname = os.path.normpath(q['content'])
            if not os.path.isfile(new_fname):
                err_msg = f'The query {i} = {q} is wrong, because the image file "{new_fname}" does not exist.'
                conversation_logger.error(err_msg)
                raise ValueError(err_msg)
            image_file_list.append(new_fname)
        else:
            new_fname = os.path.normpath(q['content'])
            if not os.path.isfile(new_fname):
                err_msg = f'The query {i} = {q} is wrong, because the image file "{new_fname}" does not exist.'
                conversation_logger.error(err_msg)
                raise ValueError(err_msg)
            audio_file_list.append(new_fname)
    if (len(text_list) == 0) and (len(image_file_list) == 0) and (len(audio_file_list) == 0):
        err_msg = f'The query list {cur_query_list} is empty!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    return text_list, image_file_list, audio_file_list


def generate_full_prompt(model: MultimodalModel,
                         cur_query_list: List[Dict[str, str]], history_list: Tuple[str, str],
                         search_k: int = -1) -> str:
    text_list, image_file_list, audio_file_list = parse_query(cur_query_list)
    previous_dialogue, last_answer = history_list
    if len(previous_dialogue) == 0:
        if len(last_answer) > 0:
            err_msg = (f'The dialogue history {previous_dialogue} is empty, '
                       f'but the model\'s last answer {last_answer} is non empty. It is impossible!')
            conversation_logger.error(err_msg)
            raise ValueError(err_msg)
        new_prompt = config.initial_promt
    else:
        if len(last_answer) > 0:
            new_prompt = previous_dialogue + ' ' + last_answer + '</s> [INST]'
        else:
            if previous_dialogue.endswith('[/INST]'):
                new_prompt = previous_dialogue[:-7].strip()
            else:
                new_prompt = previous_dialogue
    image_descriptions = [find_text_by_image(cur, model, search_k=search_k) for cur in image_file_list]
    audio_descriptions = [find_text_by_audio(cur, model, search_k=search_k) for cur in audio_file_list]
    del audio_file_list
    if len(image_descriptions) > 0:
        new_prompt += (' ' + generate_prompt_for_image(image_descriptions))
    if len(audio_descriptions) > 0:
        new_prompt += (' ' + generate_prompt_for_audio(audio_descriptions))
    for cur_text in text_list:
        new_prompt += (' ' + cur_text)
    del text_list
    new_prompt += ' [/INST]'
    return ' '.join(new_prompt.strip().split())


"""
Main functions
"""


def setup_model_and_tokenizer() -> Tuple[MultimodalModel, AutoTokenizer]:
    # if not torch.cuda.is_available():
    #     err_msg = 'CUDA is not available!'
    #     conversation_logger.error(err_msg)
    #     raise ValueError(err_msg)
    one_peace_dir = os.path.join(os.path.dirname(__file__), 'ONE-PEACE')
    # if not os.path.isdir(one_peace_dir):
    #     err_msg = f'The directory "{one_peace_dir}" does not exist!'
    #     conversation_logger.error(err_msg)
    #     raise ValueError(err_msg)
    # sys.path.append(one_peace_dir)
    # from one_peace.models import from_pretrained

    # conversation_logger.info('ONE-PEACE is attached.')

    # model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_dir = config.weights_path
    if not os.path.isdir(model_dir):
        err_msg = f'The directory "{model_dir}" does not exist!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)

    one_peace_model_fname = os.path.join(model_dir, 'one-peace.pt')
    if not os.path.isfile(one_peace_model_fname):
        err_msg = f'The file "{one_peace_model_fname}" does not exist!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    current_workdir = os.getcwd()
    conversation_logger.info(f'Current working directory: {current_workdir}')
    if config.load_one_peace:
        os.chdir(one_peace_dir)
        conversation_logger.info(f'New working directory: {os.getcwd()}')
        if DEVICE.type == "cpu":
            onepeace_model = from_pretrained(one_peace_model_fname, device=DEVICE)
        else:
            onepeace_model = from_pretrained(one_peace_model_fname, device=DEVICE, dtype='fp16')
        conversation_logger.info(f'ONE-PEACE model is loaded from the "{one_peace_model_fname}".')
        os.chdir(current_workdir)
        conversation_logger.info(f'Restored working directory: {os.getcwd()}')
        conversation_logger.info('The ONE-PEACE model is loaded.')

        pca_fname = os.path.join(model_dir, 'wiki_onepeace_pca.pkl')
        if not os.path.isfile(pca_fname):
            err_msg = f'The file "{pca_fname}" does not exist!'
            conversation_logger.error(err_msg)
            raise ValueError(err_msg)
        with open(pca_fname, 'rb') as fp:
            pca = pickle.load(fp)
        if not isinstance(pca, Pipeline):
            err_msg = (f'The PCA pipeline loaded from the "{pca_fname}" has a wrong type! '
                    f'Expected sklearn.pipeline.Pipeline, got {type(pca)}.')
            conversation_logger.error(err_msg)
            raise ValueError(err_msg)
        feature_vector_size = pca.named_steps.pca.n_components
        conversation_logger.info(f'The PCA pipeline is loaded. The feature vector size is {feature_vector_size}.')

        texts_fname = os.path.join(model_dir, 'en_wiki_paragraphs.txt')
        if not os.path.isfile(texts_fname):
            err_msg = f'The file "{texts_fname}" does not exist!'
            conversation_logger.error(err_msg)
            raise ValueError(err_msg)
        paragraphs = []
        counter = 0
        for curline in fileinput.input(texts_fname, openhook=fileinput.hook_encoded("utf-8", "surrogateescape")):
            prepline = curline.strip()
            if len(prepline) > 0:
                paragraphs.append(prepline)
                counter += 1
                if counter % 1_000_000 == 0:
                    conversation_logger.info(f'{counter} paragraphs are loaded from the "{texts_fname}".')
        gc.collect()
        info_msg = (f'The text corpus with Wikipedia paragraphs is loaded. '
                    f'There are {len(paragraphs)} paragraphs in this corpus.')
        conversation_logger.info(info_msg)

        annoy_fname = os.path.join(model_dir, 'en_wiki_paragraphs.ann')
        if not os.path.isfile(annoy_fname):
            err_msg = f'The file "{annoy_fname}" does not exist!'
            conversation_logger.error(err_msg)
            raise ValueError(err_msg)
        annoy_index = AnnoyIndex(feature_vector_size, 'angular')
        annoy_index.load(annoy_fname)
        conversation_logger.info('The Annoy index for Wikipedia paragraphs is loaded.')
        n_annoy_items = annoy_index.get_n_items()
        if n_annoy_items != len(paragraphs):
            err_msg = (f'The Wiki text corpus does not correspond to the Wiki text index, '
                    f'because their sizes are not same! {n_annoy_items} != {len(paragraphs)}.')
            conversation_logger.error(err_msg)
            raise ValueError(err_msg)
    else:
        onepeace_model = None
        pca = None
        annoy_index = None
        paragraphs = None

    audio_cls_dirname = os.path.join(model_dir, 'auxiliary_models', 'audioset')
    if not os.path.isdir(audio_cls_dirname):
        err_msg = f'The directory "{audio_cls_dirname}" does not exist!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    audio_fe = AutoFeatureExtractor.from_pretrained(audio_cls_dirname)
    if DEVICE.type == "cpu":
        audio_cls = ASTForAudioClassification.from_pretrained(audio_cls_dirname).to(DEVICE)
    else:
        audio_cls = ASTForAudioClassification.from_pretrained(audio_cls_dirname, torch_dtype=torch.float16).to(DEVICE)
    
    if config.use_blit:
        image_captioning_dirname = os.path.join(model_dir, 'auxiliary_models', 'blip')
        if not os.path.isdir(image_captioning_dirname):
            err_msg = f'The directory "{image_captioning_dirname}" does not exist!'
            conversation_logger.error(err_msg)
            raise ValueError(err_msg)
        image_processor = BlipProcessor.from_pretrained(image_captioning_dirname)
        if DEVICE.type == "cpu":
            image_caption_generator = BlipForConditionalGeneration.from_pretrained(
                image_captioning_dirname
            ).to(DEVICE)
        else:
            image_caption_generator = BlipForConditionalGeneration.from_pretrained(
                image_captioning_dirname,
                torch_dtype=torch.float16
            ).to(DEVICE)
    else:
        image_captioning_dirname = None
        image_processor = None
        image_caption_generator = None

    # asr_dirname = os.path.join(model_dir, 'auxiliary_models', 'whisper_medium')
    # if not os.path.isdir(asr_dirname):
    #     err_msg = f'The directory "{asr_dirname}" does not exist!'
    #     conversation_logger.error(err_msg)
    #     raise ValueError(err_msg)
    asr_pipe = pipeline(
        'automatic-speech-recognition',
        model=config.weights_whisper,
        chunk_length_s=30,
        device=DEVICE
    )

    sbert_dirname = os.path.join(model_dir, 'auxiliary_models', 'sbert')
    if not os.path.isdir(sbert_dirname):
        err_msg = f'The directory "{sbert_dirname}" does not exist!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    sentence_embedder = SentenceTransformer(sbert_dirname, device=DEVICE.type)

    llm_dirname = config.llava_weights
    if not os.path.isdir(llm_dirname):
        err_msg = f'The directory "{llm_dirname}" does not exist!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)

    if DEVICE.type == "cpu":
        llm_model = LlavaNextForConditionalGeneration.from_pretrained(llm_dirname).to(DEVICE)
    else:
        llm_model = LlavaNextForConditionalGeneration.from_pretrained(llm_dirname, torch_dtype=torch.float16, device_map={"":0})

    llm_model.eval()
    llm_processor= LlavaNextProcessor.from_pretrained(llm_dirname)
    conversation_logger.info('The large language model is loaded.')

    # Load TrOCR model and processor
    if not os.path.isdir(model_dir):
        err_msg = f'The directory "{model_dir}" does not exist!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    trocr_processor = TrOCRProcessor.from_pretrained(config.weights_ocr)
    if DEVICE.type == "cpu":
        trocr_model = VisionEncoderDecoderModel.from_pretrained(config.weights_ocr).to(DEVICE)
    else:
        trocr_model = VisionEncoderDecoderModel.from_pretrained(config.weights_ocr, torch_dtype=torch.float16).to(DEVICE)
    conversation_logger.info('The Ocr model is loaded.')
    translate_ruen = pipeline("translation", model="/userspace/pva/weights/opusruen", device=DEVICE)
    translate_enru = pipeline("translation", model="/userspace/pva/weights/opusenru", device=DEVICE)
    conversation_logger.info('The Translation models are loaded.')

    full_pipeline_for_conversation = MultimodalModel(
        image=(image_processor, image_caption_generator),
        audio=(audio_fe, audio_cls),
        speech=asr_pipe,
        sbert=sentence_embedder,
        one_peace=onepeace_model,
        pca=pca,
        annoy_index=annoy_index,
        texts=paragraphs,
        ocr=(trocr_processor, trocr_model),
        llm=llm_model,
        translate_ruen=translate_ruen,
        translate_enru=translate_enru,
    )
    gc.collect()
    return full_pipeline_for_conversation, llm_processor


# Function that generates the responses for dialodues queries w.r.t. history.
def generate_text(model: MultimodalModel, processor: LlavaNextProcessor,
                  cur_query_list: List[Dict[str, str]], history_list: Tuple[str, str]) -> Tuple[str, Tuple[str, str]]:

    text_list, image_file_list, audio_file_list = parse_query(cur_query_list)
    is_russian = any(any(set("йцукенгшщзхъфывапролджэячсмитьбю") & set(text.lower())) for text in text_list)
    if is_russian:
        for query in cur_query_list:
            if 'type' in query and query['type'] == 'text':
                query["content"] = model.translate_ruen(query["content"])[0]['translation_text']
    if not config.bot_use_history:
        history_list = ('', '')
    prompt = generate_full_prompt(model, cur_query_list, history_list)
    conversation_logger.info(f'Current prompt: {prompt}')
    answer = generate_answer_based_on_prompt(prompt, image_file_list, model.llm, processor)
    conversation_logger.info(f'Answer: {answer}')

    answer = answer.replace("<image>", "image")
    prompt = prompt.replace("<image>", "image")

    if is_russian:
        ret_answer = model.translate_enru(answer)[0]['translation_text']
    else:
        ret_answer = answer

    history_list = (prompt, answer)

    return ret_answer, history_list



def get_ppl(model: MultimodalModel, processor: LlavaNextProcessor,
            cur_query_tuple: Tuple[List[Dict[str, str]], str],
            history_list: Tuple[str, str]) -> Tuple[float, Tuple[str, str]]:

    cur_query_list, text = cur_query_tuple
    text_list, image_file_list, audio_file_list = parse_query(cur_query_list)

    prompt = generate_full_prompt(model, cur_query_list, history_list)
    
    out_logits = generate_logits_based_on_prompt(prompt, image_file_list, model.llm, processor)

    dialogue_emb = processor.encode(prompt, add_special_tokens=False, return_tensors="pt")

    loss = nn.CrossEntropyLoss()

    shift_logits = out_logits[..., : -1, :].contiguous()
    labels = processor.encode(cur_query_tuple[1], add_special_tokens=False, return_tensors="pt")
    context_before_labels = torch.LongTensor([-100] * dialogue_emb.shape[1]).unsqueeze(0)
    labels = torch.cat([context_before_labels, labels], dim=1).to(llm.device)
    shift_labels = labels[..., 1:].contiguous()
    
    neg_log_likelihood = loss(shift_logits.transpose(1, 2), shift_labels)
    ppl = torch.exp2(neg_log_likelihood)
    
    return ppl.item(), dialogue_emb
    '''
    dialogue_emb = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    loss = nn.CrossEntropyLoss()

    shift_logits = out_logits[..., : -1, :].contiguous()
    labels = tokenizer.encode(cur_query_tuple[1], add_special_tokens=False, return_tensors="pt")
    context_before_labels = torch.LongTensor([-100] * dialogue_emb.shape[1]).unsqueeze(0)
    labels = torch.concat([context_before_labels, labels], dim=1).to(DEVICE)
    shift_labels = labels[..., 1:].contiguous()
    
    neg_log_likelihood = loss(shift_logits.transpose(1, 2), shift_labels)
    ppl = torch.exp2(neg_log_likelihood)
    
    return ppl.item(), dialogue_emb
    '''


def prepare_logger():
    conversation_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    conversation_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('multimodal_conversation.log')
    file_handler.setFormatter(formatter)
    conversation_logger.addHandler(file_handler)
