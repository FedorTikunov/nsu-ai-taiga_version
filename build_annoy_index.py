from argparse import ArgumentParser
import codecs
import copy
import csv
import gc
import logging
import os
import random
import sys
from typing import List, Tuple

from annoy import AnnoyIndex
import numpy as np
from tqdm import tqdm


indexing_logger = logging.getLogger(__name__)
RANDOM_SEED: int = 42


def load_csv(fname: str) -> List[Tuple[str, int, str]]:
    true_header = ['title', 'paragraph_id', 'paragraph_content']
    loaded_header = []
    line_idx = 1
    data = []
    with codecs.open(fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        csv_reader = csv.reader(fp, delimiter=',', quotechar='"')
        for row in csv_reader:
            if len(row) > 0:
                err_msg = f'File "{os.path.basename(fname)}": line {line_idx} is wrong!'
                if len(loaded_header) == 0:
                    loaded_header = copy.copy(row)
                    if loaded_header != true_header:
                        err_msg += f' Loaded header does not equal to the true header! {loaded_header} != {true_header}'
                        indexing_logger.error(err_msg)
                        raise ValueError(err_msg)
                else:
                    if len(row) != len(loaded_header):
                        err_msg += (f' The row size does not equal to the header size! '
                                    f'{len(row)} != {len(loaded_header)}')
                        indexing_logger.error(err_msg)
                        raise ValueError(err_msg)
                    article_title = row[0].strip()
                    try:
                        paragraph_id = int(row[1])
                    except:
                        paragraph_id = -1
                    if paragraph_id < 0:
                        err_msg += f' The paragraph ID = {row[1]} is wrong!'
                        indexing_logger.error(err_msg)
                        raise ValueError(err_msg)
                    paragraph_text = row[2].strip()
                    data.append((article_title, paragraph_id, paragraph_text))
            line_idx += 1
    return data


def load_numpy_array(fname: str, feature_vector_size: int = None) -> Tuple[np.ndarray, int]:
    data = np.load(fname, allow_pickle=False)
    if not isinstance(data, np.ndarray):
        err_msg = f'The file "{fname}" contains a wrong data! Expected {type(np.array([1, 2]))}, got {type(data)}.'
        indexing_logger.error(err_msg)
        raise ValueError(err_msg)
    if len(data.shape) != 2:
        err_msg = f'The file "{fname}" contains a wrong data! Expected 2-D array, got {len(data.shape)}-D one.'
        indexing_logger.error(err_msg)
        raise ValueError(err_msg)
    if feature_vector_size is None:
        feature_vector_size = data.shape[1]
    elif feature_vector_size != data.shape[1]:
        err_msg = (f'The file "{fname}" contains a wrong data! Feature vector size is unexpected. '
                   f'Expected {feature_vector_size}, got {data.shape[1]}.')
        indexing_logger.error(err_msg)
        raise ValueError(err_msg)
    return data, feature_vector_size


def match_files(csv_dirname: str, np_dirname: str) -> List[Tuple[str, str]]:
    csv_files = sorted(list(map(
        lambda it2: os.path.join(csv_dirname, it2),
        filter(
            lambda it1: it1.endswith('.csv'), os.listdir(csv_dirname)
        )
    )))
    if len(csv_files) == 0:
        err_msg = f'The directory "{csv_dirname}" is empty!'
        indexing_logger.error(err_msg)
        raise ValueError(err_msg)
    indexing_logger.info(f'There are {len(csv_files)} CSV files in the directory "{csv_dirname}".')
    numpy_files = sorted(list(map(
        lambda it2: os.path.join(csv_dirname, it2),
        filter(
            lambda it1: it1.endswith('.npy'), os.listdir(np_dirname)
        )
    )))
    if len(numpy_files) == 0:
        err_msg = f'The directory "{np_dirname}" is empty!'
        indexing_logger.error(err_msg)
        raise ValueError(err_msg)
    indexing_logger.info(f'There are {len(numpy_files)} CSV files in the directory "{np_dirname}".')
    intersection = (set(map(lambda x: os.path.basename(x)[:-4], csv_files)) &
                    set(map(lambda y: os.path.basename(y)[:-4], numpy_files)))
    if len(intersection) == 0:
        err_msg = 'There is an empty intersection between CSV and NumPy data files!'
        indexing_logger.error(err_msg)
        raise ValueError(err_msg)
    intersection = sorted(list(intersection))
    res = []
    for base_fname in intersection:
        res.append((os.path.join(csv_dirname, base_fname + '.csv'), os.path.join(np_dirname, base_fname + '.npy')))
    indexing_logger.info(f'There are {len(res)} file pairs CSV-NumPy.')
    return res


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-c', '--csv', dest='csv_wiki_dir', type=str, required=True,
                        help='The input path to the directory with CSV files containing tokenized Wikipedia.')
    parser.add_argument('-n', '--numpy', dest='numpy_wiki_dir', type=str, required=True,
                        help='The input path to the directory with CSV files containing vectorized Wikipedia.')
    parser.add_argument('-i', '--index', dest='annoy_index_dir', type=str, required=True,
                        help='The directory into which the Annoy index and the text collection will be saved.')
    parser.add_argument('--n_trees', dest='annoy_trees', type=int, required=False, default=10,
                        help='The number of trees for indexing.')
    args = parser.parse_args()

    csv_data_dir = os.path.normpath(args.csv_wiki_dir)
    if not os.path.isdir(csv_data_dir):
        err_msg = f'The directory "{csv_data_dir}" does not exist!'
        indexing_logger.error(err_msg)
        raise IOError(err_msg)

    np_data_dir = os.path.normpath(args.numpy_wiki_dir)
    if not os.path.isdir(np_data_dir):
        err_msg = f'The directory "{np_data_dir}" does not exist!'
        indexing_logger.error(err_msg)
        raise IOError(err_msg)

    annoy_index_dir = os.path.normpath(args.annoy_index_dir)
    if not os.path.isdir(annoy_index_dir):
        err_msg = f'The directory "{annoy_index_dir}" does not exist!'
        indexing_logger.error(err_msg)
        raise IOError(err_msg)

    file_pairs = match_files(csv_data_dir, np_data_dir)
    feature_vector_size = None
    annoy_index = None
    text_collection = []
    for csv_fname, np_fname in tqdm(file_pairs):
        texts = load_csv(csv_fname)
        vectors, feature_vector_size = load_numpy_array(np_fname, feature_vector_size)
        if len(texts) != vectors.shape[0]:
            err_msg = (f'{os.path.basename(csv_fname)[:-4]}: the number of text = {len(texts)} does not equal to '
                       f'the number of text vectors = {vectors.shape[0]}!')
            indexing_logger.error(err_msg)
            raise ValueError(err_msg)
        if annoy_index is None:
            annoy_index = AnnoyIndex(feature_vector_size, 'angular')
        for i in range(vectors.shape[0]):
            annoy_index.add_item(len(text_collection), vectors[i])
            text_collection.append(texts[i][2])
        del texts, vectors
        gc.collect()

    text_collection_fname = os.path.join(annoy_index_dir, 'en_wiki_paragraphs.txt')
    with codecs.open(text_collection_fname, mode='w', encoding='utf-8') as fp:
        for cur in text_collection:
            fp.write(cur + '\n')
    del text_collection
    gc.collect()
    indexing_logger.info(f'The text collection has been saved into the "{text_collection_fname}".')

    annoy_index_fname = os.path.join(annoy_index_dir, 'en_wiki_paragraphs.ann')
    annoy_index.set_seed(RANDOM_SEED)
    annoy_index.build(args.annoy_trees, n_jobs=-1)
    annoy_index.save(annoy_index_fname)
    indexing_logger.info(f'The Annoy index has been saved into the "{annoy_index_fname}".')


if __name__ == '__main__':
    indexing_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    indexing_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('wikipedia_annoy_indexing.log')
    file_handler.setFormatter(formatter)
    indexing_logger.addHandler(file_handler)
    main()
