import codecs
import pickle
from collections import namedtuple
import gc
import logging
import os
import sys
from typing import Dict, List, Tuple

from annoy import AnnoyIndex
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# DEVICE = torch.device("cuda:0")
DEVICE = torch.device("cpu")

MultimodalModel = namedtuple('MultimodalModel', 'one_peace pca annoy_index texts llm')
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


def find_texts_by_image(image_fname: str, model: MultimodalModel, top_n: int = 1, search_k: int = -1) -> List[str]:
    if not os.path.isfile(image_fname):
        err_msg = f'The image "{image_fname}" does not exist!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    src_images = model.one_peace.process_image([image_fname])
    with torch.no_grad():
        image_features = model.one_peace.extract_image_features(src_images)
    del src_images
    image_vector = model.pca.transform(image_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
    del image_features
    found_indices = model.annoy_index.get_nns_by_vector(image_vector, n=top_n, search_k=search_k)
    del image_vector
    return [model.texts[idx] for idx in found_indices]


def find_texts_by_audio(audio_fname: str, model: MultimodalModel, top_n: int = 1, search_k: int = -1) -> List[str]:
    if not os.path.isfile(audio_fname):
        err_msg = f'The image "{audio_fname}" does not exist!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    src_audios, audio_padding_masks = model.one_peace.process_audio([audio_fname])
    with torch.no_grad():
        audio_features = model.one_peace.extract_audio_features(src_audios, audio_padding_masks)
    del src_audios, audio_padding_masks
    audio_vector = model.pca.transform(audio_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
    del audio_features
    found_indices = model.annoy_index.get_nns_by_vector(audio_vector, n=top_n, search_k=search_k)
    del audio_vector
    return [model.texts[idx] for idx in found_indices]


def tokenize_prompt(prompt: str, tokenizer: AutoTokenizer, add_eos_token: bool = True,
                    add_labels: bool=True) -> Dict[str, List[int]]:
    result = tokenizer(prompt, padding=False, return_tensors=None)
    if (result['input_ids'][-1] != tokenizer.eos_token_id) and add_eos_token:
        result['input_ids'].append(tokenizer.eos_token_id)
        result['attention_mask'].append(1)
    if add_labels:
        result['labels'] = result['input_ids'].copy()
    return result


def generate_answer_based_on_prompt(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> str:
    tokenized_text = tokenize_prompt(
        prompt,
        tokenizer,
        add_labels=False
    )
    input_ids = [torch.tensor(data=tokenized_text['input_ids'], dtype=torch.long)]
    attention_mask = [torch.tensor(data=tokenized_text['attention_mask'], dtype=torch.long)]
    del tokenized_text
    batched_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True, padding_value=0  # <unk> idx
    ).to(DEVICE)
    batched_attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask,
        batch_first=True, padding_value=0
    ).to(DEVICE)
    generated_ids = model.generate(
        input_ids=batched_input_ids, attention_mask=batched_attention_mask,
        max_new_tokens=1000, do_sample=True
    )
    del batched_input_ids, batched_attention_mask
    predicted_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    input_prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    del input_ids, attention_mask, generated_ids
    if len(predicted_text) < len(input_prompt):
        err_msg = (f'The predicted answer "{predicted_text}" does not correct, '
                   f'because it does not start with the prompt "{input_prompt}".')
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    if not predicted_text.startswith(input_prompt):
        err_msg = (f'The predicted answer "{predicted_text}" does not correct, '
                   f'because it does not start with the prompt "{input_prompt}".')
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    return ' '.join(predicted_text[len(input_prompt):].split()).strip()


def generate_logits_based_on_prompt(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    tokenized_text = tokenize_prompt(
        prompt,
        tokenizer,
        add_labels=False
    )
    input_ids = [torch.tensor(data=tokenized_text['input_ids'], dtype=torch.long)]
    attention_mask = [torch.tensor(data=tokenized_text['attention_mask'], dtype=torch.long)]
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
            return_dict=True
        ).logits

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
        prompt = (f'Please imagine that you have just looked at an image that probably corresponds to '
                  f'the following text description. {image_description[0]}')
        if prompt[-1] not in PUNCTUATION:
            prompt += '.'
        elif prompt[-1] not in {'.', '?', '!'}:
            prompt = prompt[:-1] + '.'
    else:
        prompt = (f'Please imagine that you have just looked at {cardinal_to_str(len(image_description))} images '
                  f'that probably match the following text descriptions.')
        for k, v in enumerate(image_description):
            prompt += f' The {ordinal_to_str(k + 1)} image. {v}'
            if prompt[-1] not in PUNCTUATION:
                prompt += '.'
            elif prompt[-1] not in {'.', '?', '!'}:
                prompt = prompt[:-1] + '.'
    return prompt


def generate_prompt_for_audio(audio_description: List[str]) -> str:
    if len(audio_description) < 1:
        return ''
    if len(audio_description) == 1:
        prompt = (f'Please imagine that you have just heard a sound that probably corresponds to '
                  f'the following text description. {audio_description[0]}')
        if prompt[-1] not in PUNCTUATION:
            prompt += '.'
        elif prompt[-1] not in {'.', '?', '!'}:
            prompt = prompt[:-1] + '.'
    else:
        prompt = (f'Please imagine that you have just heard {cardinal_to_str(len(audio_description))} sounds '
                  f'that probably match the following text descriptions.')
        for k, v in enumerate(audio_description):
            prompt += f' The {ordinal_to_str(k + 1)} sound. {v}'
            if prompt[-1] not in PUNCTUATION:
                prompt += '.'
            elif prompt[-1] not in {'.', '?', '!'}:
                prompt = prompt[:-1] + '.'
    return prompt


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
        new_prompt = ('<s>[INST] You are a useful and friendly interlocutor with great erudition and '
                      'developed intelligence. You can keep up a conversation on various topics and even know '
                      'how to play complex intellectual games. ')
    else:
        if len(last_answer) > 0:
            new_prompt = previous_dialogue + ' ' + last_answer + '</s> [INST]'
        else:
            if previous_dialogue.endswith('[/INST]'):
                new_prompt = previous_dialogue[:-7].strip()
            else:
                new_prompt = previous_dialogue
    for cur_text in text_list:
        new_prompt += (' ' + cur_text)
    image_descriptions = [find_texts_by_image(cur, model, search_k=search_k)[0] for cur in image_file_list]
    audio_descriptions = [find_texts_by_audio(cur, model, search_k=search_k)[0] for cur in audio_file_list]
    del text_list, image_file_list, audio_file_list
    if len(image_descriptions) > 0:
        new_prompt += (' ' + generate_prompt_for_image(image_descriptions))
    if len(audio_descriptions) > 0:
        new_prompt += (' ' + generate_prompt_for_audio(audio_descriptions))
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
    if not os.path.isdir(one_peace_dir):
        err_msg = f'The directory "{one_peace_dir}" does not exist!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    sys.path.append(one_peace_dir)
    from one_peace.models import from_pretrained

    conversation_logger.info('ONE-PEACE is attached.')

    model_dir = os.path.join(os.path.dirname(__file__), 'models')
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
    with codecs.open(texts_fname, mode='r', encoding='utf-8') as fp:
        paragraphs = list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), fp.readlines())))
    gc.collect()
    conversation_logger.info('The text corpus with Wikipedia paragraphs is loaded.')

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

    llm_dirname = os.path.join(model_dir, 'llm')
    if not os.path.isdir(llm_dirname):
        err_msg = f'The directory "{llm_dirname}" does not exist!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)

    if DEVICE.type == "cpu":
        llm_model = AutoModelForCausalLM.from_pretrained(llm_dirname).to(DEVICE)
    else:
        llm_model = AutoModelForCausalLM.from_pretrained(llm_dirname, torch_dtype=torch.float16).to(DEVICE)

    llm_model.eval()
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_dirname)
    conversation_logger.info('The large language model is loaded.')

    full_pipeline_for_conversation = MultimodalModel(
        one_peace=onepeace_model,
        pca=pca,
        annoy_index=annoy_index,
        texts=paragraphs,
        llm=llm_model
    )
    gc.collect()
    return full_pipeline_for_conversation, llm_tokenizer


# Function that generates the responses for dialodues queries w.r.t. history.
def generate_text(model: MultimodalModel, tokenizer: AutoTokenizer,
                  cur_query_list: List[Dict[str, str]], history_list: Tuple[str, str]) -> Tuple[str, Tuple[str, str]]:

    prompt = generate_full_prompt(model, cur_query_list, history_list)
    answer = generate_answer_based_on_prompt(prompt, model.llm, tokenizer)

    history_list = (prompt, answer)

    return answer, history_list


def get_ppl(model: MultimodalModel, tokenizer: AutoTokenizer,
            cur_query_tuple: Tuple[List[Dict[str, str]], str],
            history_list: Tuple[str, str]) -> Tuple[float, Tuple[str, str]]:

    cur_query_list, text = cur_query_tuple

    prompt = generate_full_prompt(model, cur_query_list, history_list)
    out_logits = generate_logits_based_on_prompt(prompt, model.llm, tokenizer)

    dialogue_emb = tokenizer.encode(prompt)

    loss = nn.CrossEntropyLoss()

    shift_logits = out_logits[..., : -1, :].contiguous()
    labels = tokenizer.encode(cur_query_tuple[1], add_special_tokens=False, return_tensors="pt")
    context_before_labels = torch.LongTensor([-100] * dialogue_emb.shape[1]).unsqueeze(0)
    labels = torch.concat([context_before_labels, labels], dim=1).to(DEVICE)
    shift_labels = labels[..., 1:].contiguous()
    
    neg_log_likelihood = loss(shift_logits.transpose(1, 2), shift_labels)
    ppl = torch.exp2(neg_log_likelihood)
    
    return ppl.item(), dialogue_emb
