import json
from argparse import ArgumentParser
import codecs
import logging
import os
import pickle
import sys

from annoy import AnnoyIndex
import torch


object_search_logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_object', type=str, required=True,
                        help='The path to the input object (image or audio).')
    parser.add_argument('-t', '--type', dest='input_object_type', type=str, required=True, choices=['image', 'audio'],
                        help='The input object type (image or audio).')
    parser.add_argument('-o', '--output', dest='output_json', type=str, required=True,
                        help='The path to the output JSON file with object descriptions ordered by similarity.')
    parser.add_argument('-n', '--top_n', dest='top_n', type=int, required=False, default=1,
                        help='Top N of the most similar object descriptions.')
    parser.add_argument('-p', '--one-peace', dest='one_peace_dir', type=str, required=True,
                        help='The path to ONE-PEACE repository.')
    parser.add_argument('-a', '--annoy', dest='annoy_index_path', type=str, required=True,
                        help='The path to the directory with Annoy index, text corpus and PCA pipeline.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The model name.')
    parser.add_argument('--dtype', dest='torch_dtype', type=str, required=False, default='fp16',
                        help='The model data type.')
    parser.add_argument('--device', dest='torch_device', type=str, required=False, default=None,
                        help='The model device.', choices=['cuda', 'cpu', 'gpu'])
    args = parser.parse_args()

    if args.torch_device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        if args.torch_device not in {'cpu', 'cuda', 'gpu'}:
            err_msg = f'The device "{args.torch_device}" is unknown!'
            object_search_logger.error(err_msg)
            raise ValueError(err_msg)
        if (not torch.cuda.is_available()) and (args.torch_device in {'cuda', 'gpu'}):
            err_msg = f'The device "{args.torch_device}" is not available!'
            object_search_logger.error(err_msg)
            raise ValueError(err_msg)
        device = 'cpu' if (args.torch_device == 'cpu') else 'cuda'
    object_search_logger.info(f'{device.upper()} is used.')

    input_object_fname = os.path.normpath(args.input_object)
    if not os.path.isfile(input_object_fname):
        err_msg = f'The file "{input_object_fname}" does not exist!'
        object_search_logger.error(err_msg)
        raise ValueError(err_msg)

    annoy_index_dir = args.annoy_index_path
    if not os.path.isdir(annoy_index_dir):
        err_msg = f'The directory "{annoy_index_dir}" does not exist!'
        object_search_logger.error(err_msg)
        raise ValueError(err_msg)
    annoy_index_fname = os.path.join(annoy_index_dir, 'en_wiki_paragraphs.ann')
    text_corpus_fname = os.path.join(annoy_index_dir, 'en_wiki_paragraphs.txt')
    pca_pipeline_fname = os.path.join(annoy_index_dir, 'wiki_onepeace_pca.pkl')
    if not os.path.isfile(annoy_index_fname):
        err_msg = f'The file "{annoy_index_fname}" does not exist!'
        object_search_logger.error(err_msg)
        raise ValueError(err_msg)
    if not os.path.isfile(text_corpus_fname):
        err_msg = f'The file "{text_corpus_fname}" does not exist!'
        object_search_logger.error(err_msg)
        raise ValueError(err_msg)
    if not os.path.isfile(pca_pipeline_fname):
        err_msg = f'The file "{pca_pipeline_fname}" does not exist!'
        object_search_logger.error(err_msg)
        raise ValueError(err_msg)

    output_json_fname = os.path.normpath(args.output_json)
    output_json_dir = os.path.dirname(output_json_fname)
    if len(output_json_dir) > 0:
        if not os.path.isdir(output_json_dir):
            err_msg = f'The directory "{output_json_dir}" does not exist!'
            object_search_logger.error(err_msg)
            raise ValueError(err_msg)
    if os.path.basename(output_json_fname) == os.path.basename(input_object_fname):
        err_msg = 'The input and output files have same names!'
        object_search_logger.error(err_msg)
        raise ValueError(err_msg)

    one_peace_dir = os.path.normpath(args.one_peace_dir)
    if not os.path.isdir(one_peace_dir):
        err_msg = f'The directory "{one_peace_dir}" does not exist!'
        object_search_logger.error(err_msg)
        raise ValueError(err_msg)

    model_name = os.path.normpath(args.model_name)
    if not os.path.isfile(model_name):
        err_msg = f'The file "{model_name}" does not exist!'
        object_search_logger.error(err_msg)
        raise ValueError(err_msg)

    with open(pca_pipeline_fname, 'rb') as fp:
        pca_pipeline = pickle.load(fp)

    sys.path.append(os.path.join(one_peace_dir))
    from one_peace.models import from_pretrained
    object_search_logger.info('ONE-PEACE is attached.')

    current_workdir = os.getcwd()
    object_search_logger.info(f'Current working directory: {current_workdir}')
    os.chdir(one_peace_dir)
    object_search_logger.info(f'New working directory: {os.getcwd()}')
    model = from_pretrained(model_name, device=device, dtype=args.torch_dtype)
    object_search_logger.info('Model is loaded.')
    os.chdir(current_workdir)
    object_search_logger.info(f'Restored working directory: {os.getcwd()}')

    if args.input_object_type == 'image':
        src_images = model.process_image([input_object_fname])
        with torch.no_grad():
            image_features = model.extract_image_features(src_images)
        object_vector = pca_pipeline.transform(image_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
        object_search_logger.info('The input image is vectorized.')
    else:
        src_audios, audio_padding_masks = model.one_peace.process_audio([input_object_fname])
        with torch.no_grad():
            audio_features = model.extract_audio_features(src_audios, audio_padding_masks)
        object_vector = pca_pipeline.transform(audio_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
        object_search_logger.info('The input audio is vectorized.')
    feature_vector_size = object_vector.shape[0]
    object_search_logger.info(f'The feature vector size is {feature_vector_size}.')

    annoy_index = AnnoyIndex(feature_vector_size, 'angular')
    annoy_index.load(annoy_index_fname)
    n_annoy_items = annoy_index.get_n_items()
    info_msg = f'The Annoy index for Wikipedia paragraphs is loaded. There are {n_annoy_items} items in this index.'
    object_search_logger.info(info_msg)

    paragraphs = []
    counter = 0
    with codecs.open(text_corpus_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        for curline in fp:
            prepline = curline.strip()
            if len(prepline) > 0:
                paragraphs.append(prepline)
                counter += 1
                if counter % 1_000_000 == 0:
                    object_search_logger.info(f'{counter} paragraphs are loaded from the "{text_corpus_fname}".')
    if n_annoy_items != len(paragraphs):
        err_msg = (f'The Wiki text corpus does not correspond to the Wiki text index, '
                   f'because their sizes are not same! {n_annoy_items} != {len(paragraphs)}.')
        object_search_logger.error(err_msg)
        raise ValueError(err_msg)
    object_search_logger.info('All Wikipedia paragraphs are loaded.')

    found_indices, distances = model.annoy_index.get_nns_by_vector(
        object_vector,
        n=args.top_n, search_k=-1, include_distances=True
    )
    object_search_logger.info(f'{len(found_indices)} paragraphs are found in the Annoy index.')
    found_paragraphs = [paragraphs[idx] for idx in found_indices]
    res = []
    for cur_paragraph, cur_distance in zip(found_paragraphs, distances):
        res.append({
            'text': cur_paragraph,
            'dist': float(cur_distance)
        })
    with codecs.open(output_json_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
        json.dump(obj=res, fp=fp, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    object_search_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    object_search_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('object_search.log')
    file_handler.setFormatter(formatter)
    object_search_logger.addHandler(file_handler)
    main()
