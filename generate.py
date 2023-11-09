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
from transformers import AutoModelForCausalLM, AutoTokenizer


MultimodalModel = namedtuple('MultimodalModel', 'one_peace pca annoy_index texts llm')
conversation_logger = logging.getLogger(__name__)


def setup_model_and_tokenizer() -> Tuple[MultimodalModel, AutoTokenizer]:
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
    onepeace_model = from_pretrained(one_peace_model_fname, device='cuda', dtype='fp16')
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

    llm_dirname = os.path.join(model_dir, 'llm')
    if not os.path.isdir(llm_dirname):
        err_msg = f'The directory "{llm_dirname}" does not exist!'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)

    llm_model = AutoModelForCausalLM.from_pretrained(llm_dirname, torch_dtype='fp16').cuda()
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


def find_text_for_image():
    pass


def generate_text(model: MultimodalModel, tokenizer: AutoTokenizer,
                  cur_query_list: List[Dict], history_list: Tuple[object, str]) -> Tuple[str, object]:
    pass
