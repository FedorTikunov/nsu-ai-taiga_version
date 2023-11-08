from argparse import ArgumentParser
import codecs
import copy
import csv
import gc
import logging
import math
import os
import pickle
import random
import sys
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch
from tqdm import trange


vectorization_logger = logging.getLogger(__name__)
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
                        vectorization_logger.error(err_msg)
                        raise ValueError(err_msg)
                else:
                    if len(row) != len(loaded_header):
                        err_msg += (f' The row size does not equal to the header size! '
                                    f'{len(row)} != {len(loaded_header)}')
                        vectorization_logger.error(err_msg)
                        raise ValueError(err_msg)
                    article_title = row[0].strip()
                    try:
                        paragraph_id = int(row[1])
                    except:
                        paragraph_id = -1
                    if paragraph_id < 0:
                        err_msg += f' The paragraph ID = {row[1]} is wrong!'
                        vectorization_logger.error(err_msg)
                        raise ValueError(err_msg)
                    paragraph_text = row[2].strip()
                    data.append((article_title, paragraph_id, paragraph_text))
            line_idx += 1
    return data


def main():
    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        vectorization_logger.error(err_msg)
        raise ValueError(err_msg)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.random.manual_seed(RANDOM_SEED)
    torch.cuda.random.manual_seed_all(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_wiki_dir', type=str, required=True,
                        help='The input path to the directory with CSV files containing tokenized Wikipedia.')
    parser.add_argument('-o', '--output', dest='output_csv', type=str, required=True,
                        help='The output path to the directory into which vectorized data will be saved.')
    parser.add_argument('-p', '--one-peace', dest='one_peace_dir', type=str, required=True,
                        help='The path to ONE-PEACE repository.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The ONE-PEACE model name.')
    parser.add_argument('--dtype', dest='torch_dtype', type=str, required=False, default='fp16',
                        choices=['fp16', 'bf16', 'float32'], help='The ONE-PEACE model data type.')
    parser.add_argument('--minibatch', dest='minibatch', type=int, required=False, default=16,
                        help='The mini-batch size.')
    args = parser.parse_args()

    input_data_dir = os.path.normpath(args.input_wiki_dir)
    if not os.path.isdir(input_data_dir):
        err_msg = f'The directory "{input_data_dir}" does not exist!'
        vectorization_logger.error(err_msg)
        raise IOError(err_msg)

    output_data_dir = os.path.normpath(args.output_csv)
    if not os.path.isdir(output_data_dir):
        err_msg = f'The directory "{output_data_dir}" does not exist!'
        vectorization_logger.error(err_msg)
        raise IOError(err_msg)

    one_peace_dir = os.path.normpath(args.one_peace_dir)
    if not os.path.isdir(one_peace_dir):
        err_msg = f'The directory "{one_peace_dir}" does not exist!'
        vectorization_logger.error(err_msg)
        raise IOError(err_msg)

    model_name = os.path.normpath(args.model_name)
    if not os.path.isfile(model_name):
        err_msg = f'The file "{model_name}" does not exist!'
        vectorization_logger.error(err_msg)
        raise IOError(err_msg)

    sys.path.append(os.path.join(one_peace_dir))
    from one_peace.models import from_pretrained
    vectorization_logger.info('ONE-PEACE is attached.')

    current_workdir = os.getcwd()
    vectorization_logger.info(f'Current working directory: {current_workdir}')
    os.chdir(one_peace_dir)
    vectorization_logger.info(f'New working directory: {os.getcwd()}')
    model = from_pretrained(model_name, device='cuda', dtype=args.torch_dtype)
    vectorization_logger.info('Model is loaded.')
    os.chdir(current_workdir)
    vectorization_logger.info(f'Restored working directory: {os.getcwd()}')

    csv_names = sorted(
        list(map(
            lambda it2: os.path.join(input_data_dir, it2),
            filter(
                lambda it1: it1.lower().endswith('.csv'),
                os.listdir(input_data_dir)
            )
        )),
        key=lambda it3: (-math.ceil(os.path.getsize(it3) / 1024), it3)
    )
    if len(csv_names) == 0:
        err_msg = f'The directory "{input_data_dir}" is empty!'
        vectorization_logger.error(err_msg)
        raise IOError(err_msg)

    pca = None

    for csv_counter, csv_fname in enumerate(csv_names):
        vectorization_logger.info(f'Processing of the {csv_counter + 1} file of {len(csv_names)} has begun.')
        data = load_csv(csv_fname)
        info_msg = f'Processing of the {csv_counter + 1} file of {len(csv_names)}: ' \
                   f'{len(data)} samples have been loaded from "{csv_fname}".'
        vectorization_logger.info(info_msg)
        texts = [cur[2] for cur in data]
        del data
        n_batches = math.ceil(len(texts) / args.minibatch)
        text_vectors = []
        for batch_idx in trange(n_batches):
            batch_start = batch_idx * args.minibatch
            batch_end = min(len(texts), batch_start + args.minibatch)
            src_tokens = model.process_text(texts[batch_start:batch_end])
            with torch.no_grad():
                text_features = model.extract_text_features(src_tokens)
            text_vectors.append(text_features.cpu().type(torch.FloatTensor).numpy())
            del text_features, src_tokens
        text_vectors = np.vstack(text_vectors)
        info_msg = (f'Processing of the {csv_counter + 1} file of {len(csv_names)}: '
                    f'{text_vectors.shape[0]} samples have been vectorized. '
                    f'The vector size is {text_vectors.shape[1]}.')
        vectorization_logger.info(info_msg)
        if pca is None:
            pca = Pipeline(steps=[
                ('mean', StandardScaler(with_mean=True, with_std=False)),
                ('pca', PCA(n_components=300, random_state=RANDOM_SEED)),
                ('std', StandardScaler(with_mean=True, with_std=True))
            ])
            pca.fit(text_vectors)
            pca_fname = os.path.join(output_data_dir, 'wiki_onepeace_pca.pkl')
            with open(pca_fname, 'wb') as fp:
                pickle.dump(pca, fp)
            info_msg = (f'PCA has been trained for dimensionality reduction, '
                        f'and it is saved into the file "{pca_fname}".')
            vectorization_logger.info(info_msg)
        text_vectors = pca.transform(text_vectors)
        vectorization_logger.info('The text vector size has been reduced with PCA.')
        binary_fname = os.path.join(output_data_dir, os.path.basename(csv_fname)[:-4] + '.npy')
        np.save(file=binary_fname, arr=text_vectors, allow_pickle=False)
        info_msg = f'Processing of the {csv_counter + 1} file of {len(csv_names)}: ' \
                   f'{text_vectors.shape[0]} samples have been saved into "{binary_fname}".'
        vectorization_logger.info(info_msg)
        del text_vectors, texts
        gc.collect()


if __name__ == '__main__':
    vectorization_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    vectorization_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('wikipedia_vectorization.log')
    file_handler.setFormatter(formatter)
    vectorization_logger.addHandler(file_handler)
    main()
