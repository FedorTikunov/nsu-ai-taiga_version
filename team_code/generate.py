import logging
import os
from typing import Dict, List, Tuple
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
import torch
import torch.nn as nn
from config import runtime_config, startup_config
from team_code.image import load_images, generate_image_caption, extract_text_with_trocr, detect_and_crop_objects, yolo_get_wiki_text, get_wiki_texts
from team_code.text import tokenize_prompt
from team_code.audio import find_text_by_audio
from team_code.model import MultimodalModel
from pathlib import Path
from PIL.ImageFile import ImageFile

conversation_logger = logging.getLogger(__name__)


def generate_answer_based_on_prompt(prompt: str, model: LlavaNextForConditionalGeneration, processor: LlavaNextProcessor, image_file_list: List[ImageFile] = []) -> str:
        
    if startup_config.llm_type == "llava":
        images = load_images(image_file_list)
        
        inputs = processor(text=prompt, images=images, return_tensors="pt").to(startup_config.DEVICE)
        generated_ids = model.generate(**inputs, max_new_tokens=1000, do_sample=True)

        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        input_prompt = processor.decode(inputs["input_ids"][0], skip_special_tokens=True)
    else:
        tokenized_text = tokenize_prompt(
            prompt,
            processor,
            add_labels=False
        )
        input_ids = [torch.tensor(data=tokenized_text['input_ids'], dtype=torch.long)]
        attention_mask = [torch.tensor(data=tokenized_text['attention_mask'], dtype=torch.long)]
        del tokenized_text
        batched_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True, padding_value=0  # <unk> idx
        ).to(startup_config.DEVICE)[:,:-1]
        batched_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True, padding_value=0
        ).to(startup_config.DEVICE)[:,:-1]
        generated_ids = model.generate(
            input_ids=batched_input_ids, attention_mask=batched_attention_mask,
            max_new_tokens=1000, do_sample=True
        )
        del batched_input_ids, batched_attention_mask
        
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        input_prompt = processor.batch_decode(input_ids, skip_special_tokens=True)[0]
        
        del input_ids, attention_mask, generated_ids

        
    if len(predicted_text) < len(input_prompt):
        err_msg = (f'The predicted answer "{predicted_text}" does not correct, '
                f'because it does not start with the prompt "{input_prompt}".')
        raise ValueError(err_msg)
    if not predicted_text.startswith(input_prompt):
        err_msg = (f'The predicted answer "{predicted_text}" does not correct, '
                f'because it does not start with the prompt "{input_prompt}".')
        raise ValueError(err_msg)

    return ' '.join(predicted_text[len(input_prompt):].split()).strip()


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
            image_path = Path(q['content'])
            if not image_path.is_file():
                err_msg = f'The query {i} = {q} is wrong, because the image file "{image_path}" does not exist.'
                conversation_logger.error(err_msg)
                raise ValueError(err_msg)
            image_file_list.append(image_path)
        else:
            audio_path = Path(q['content'])
            if not audio_path.is_file():
                err_msg = f'The query {i} = {q} is wrong, because the image file "{audio_path}" does not exist.'
                conversation_logger.error(err_msg)
                raise ValueError(err_msg)
            audio_file_list.append(audio_path)
    if (len(text_list) == 0) and (len(image_file_list) == 0) and (len(audio_file_list) == 0):
        err_msg = f'The query list {cur_query_list} is empty!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    return text_list, image_file_list, audio_file_list


def generate_full_prompt(model: MultimodalModel,
                         text_list: List[str],
                         image_list: List[ImageFile],
                         audio_list: List[str],
                         history_list: Tuple[str, str],
                        ) -> str:
    previous_dialogue, last_answer = history_list

    concated_input = " ".join(text_list)

    images_len = len(image_list)
    empty_list = [None] * images_len
    
    # del text_list
    image_captions = empty_list
    if startup_config.load_blit and runtime_config.use_blit:
        image_captions = [generate_image_caption(cur, model) for cur in image_list]

    yolo_images, yolo_captions, yolo_probs = empty_list, empty_list, empty_list
    if startup_config.load_yolo and runtime_config.use_yolo:
        yolo_images, yolo_captions, yolo_probs = detect_and_crop_objects(image_list, model)
        if startup_config.load_blit and runtime_config.yolo_use_blip_caption:
            yolo_captions = [[generate_image_caption(crop, model) for crop in cropped_image_list] for cropped_image_list in yolo_images]

    ocr_texts = empty_list
    if startup_config.load_ocr and runtime_config.use_ocr:
        ocr_texts = [extract_text_with_trocr(cur) for cur in image_list]

    wiki_texts = empty_list
    yolo_wiki_texts = empty_list
    if startup_config.load_one_peace and runtime_config.use_one_peace:
        if runtime_config.yolo_include_one_peace:
            yolo_wiki_texts: List[List[str]] = [[yolo_get_wiki_text(caption, crop, model) for crop, caption in zip(cropped_image_list, captions)] for cropped_image_list, captions in zip(yolo_images, yolo_captions)]
        wiki_texts = [get_wiki_texts(model, concated_input, image, caption, ocr, yolo_captions, yolo_images) for image, caption, ocr, yolo_captions, yolo_images in zip(image_list, image_captions, ocr_texts, yolo_captions, yolo_images)]
        
    audio_descriptions = [find_text_by_audio(concated_input, cur, model) for cur in audio_list]

    texts = [runtime_config.initial_promt]

    for audio_description in audio_descriptions:
        if audio_description is not None:
            texts.append(runtime_config.audio_text_prefix)
            texts.append(audio_description)

    for image_caption, ocr_text, wiki_text, yolo_current_captions, yolo_wiki_texts in zip(image_captions, ocr_texts, wiki_texts, yolo_captions, yolo_wiki_texts):
        texts.append(runtime_config.image_text)
        if image_caption is not None:
            texts.append(runtime_config.image_caption_prefix)
            texts.append(image_caption)
        if wiki_text is not None:
            texts.append(runtime_config.wiki_text_prefix)
            texts.append(wiki_text)
        if ocr_text is not None:
            texts.append(runtime_config.ocr_prefix)
            texts.append(ocr_text)
        if yolo_wiki_texts:
            texts.append(runtime_config.yolo_prefix)
            for yolo_caption, yolo_wiki_text in zip(yolo_current_captions, yolo_wiki_texts):
                texts.append(runtime_config.yolo_captions_prefix)
                texts.append(yolo_caption)
                texts.append(runtime_config.wiki_yolo_texts_prefix)
                texts.append(yolo_wiki_text)
    texts.append(runtime_config.answer_postfix)

    return "; ".join(texts)

# Function that generates the responses for dialodues queries w.r.t. history.
@torch.no_grad()
def generate_text(model: MultimodalModel, processor: LlavaNextProcessor,
                  cur_query_list: List[Dict[str, str]], history_list: Tuple[str, str]) -> Tuple[str, str]:

    text_list, image_file_list, audio_file_list = parse_query(cur_query_list)
    image_file_list: List[ImageFile] = load_images(image_file_list)
    if runtime_config.use_translation:
        is_russian = any(any(set("йцукенгшщзхъфывапролджэячсмитьбю") & set(text.lower())) for text in text_list)
        if is_russian:
            for query in cur_query_list:
                if 'type' in query and query['type'] == 'text':
                    query["content"] = model.translate_ruen(query["content"])[0]['translation_text']
    if not runtime_config.bot_use_history:
        history_list = ('', '')
    prompt = generate_full_prompt(model, text_list, image_file_list, audio_file_list, history_list)
    conversation_logger.info(f'Current prompt: {prompt}')
    answer = generate_answer_based_on_prompt(prompt, model.llm, processor, image_file_list)
    conversation_logger.info(f'Answer: {answer}')

    answer = answer.replace("<image>", "image")
    prompt = prompt.replace("<image>", "image")

    if runtime_config.use_translation and is_russian:
        ret_answer = model.translate_enru(answer)[0]['translation_text']
    else:
        ret_answer = answer

    history_list = (prompt, answer)

    return ret_answer, history_list


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
    
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(startup_config.DEVICE)

    with torch.no_grad():
        logits = model(**inputs, return_dict=True).logits

    return logits.cpu().type(torch.FloatTensor)


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
    labels = torch.cat([context_before_labels, labels], dim=1).to(model.llm.device)
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
