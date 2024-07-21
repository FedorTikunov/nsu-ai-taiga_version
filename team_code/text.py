from typing import List, Dict, Tuple
import logging
from nltk import sent_tokenize
from team_code.model import MultimodalModel
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoTokenizer
import torch
from team_code.const import CARDINAL_TEENS, CARDINAL_TENS, CARDINAL_UNITS, ORDINAL_TEENS, ORDINAL_TENS, ORDINAL_UNITS


text_processing_logger = logging.getLogger(__name__)

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
    del sentence_embeddings_of_short_text, sentence_embeddings_of_long_texts
    sentences_with_distances = sorted(
        list(zip(indices_of_long_texts, distances)),
        key=lambda it: (it[1], it[0])
    )
    del distances, indices_of_long_texts
    found_idx = sentences_with_distances[0][0]
    return long_texts[found_idx]


def tokenize_prompt(prompt: str, tokenizer: AutoTokenizer, add_eos_token: bool = True,
                    add_labels: bool=True) -> Dict[str, Tuple[List[int], List[torch.Tensor]]]:
    result = tokenizer(prompt, padding=False, return_tensors=None)
    if (result['input_ids'][-1] != tokenizer.eos_token_id) and add_eos_token:
        result['input_ids'].append(tokenizer.eos_token_id)
        result['attention_mask'].append(1)
    if add_labels:
        result['labels'] = result['input_ids'].copy()
    return result


def cardinal_to_str(value: int) -> str:
    if value >= 100:
        err_msg = f'{value} is too large number of objects (images or audios)!'
        text_processing_logger.error(err_msg)
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
        text_processing_logger.error(err_msg)
        raise ValueError(err_msg)
    value_str = f'{value}'
    if len(value_str) == 1:
        return ORDINAL_UNITS[value_str]
    if value_str in ORDINAL_TEENS:
        return ORDINAL_TEENS[value_str]
    if value_str in ORDINAL_TENS:
        return ORDINAL_TENS[value_str]
    return CARDINAL_TENS[value_str[0]] + ' ' + ORDINAL_UNITS[value_str[-1]]