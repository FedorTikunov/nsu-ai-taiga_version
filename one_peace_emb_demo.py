from argparse import ArgumentParser
import logging
import os
import sys

import torch


one_peace_demo_logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--one-peace', dest='one_peace_dir', type=str, required=True,
                        help='The path to ONE-PEACE repository.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The model name.')
    parser.add_argument('--dtype', dest='torch_dtype', type=str, required=False, default='fp16',
                        help='The model name.')
    parser.add_argument('--device', dest='torch_device', type=str, required=False, default=None,
                        help='The model name.', choices=['cuda', 'cpu', 'gpu'])
    args = parser.parse_args()

    if args.torch_device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        if args.torch_device not in {'cpu', 'cuda', 'gpu'}:
            err_msg = f'The device "{args.torch_device}" is unknown!'
            one_peace_demo_logger.error(err_msg)
            raise ValueError(err_msg)
        if (not torch.cuda.is_available()) and (args.torch_device in {'cuda', 'gpu'}):
            err_msg = f'The device "{args.torch_device}" is not available!'
            one_peace_demo_logger.error(err_msg)
            raise ValueError(err_msg)
        device = 'cpu' if (args.torch_device == 'cpu') else 'cuda'
    one_peace_demo_logger.info(f'{device.upper()} is used.')

    one_peace_dir = os.path.normpath(args.one_peace_dir)
    if not os.path.isdir(one_peace_dir):
        err_msg = f'The directory "{one_peace_dir}" does not exist!'
        one_peace_demo_logger.error(err_msg)
        raise ValueError(err_msg)

    model_name = os.path.normpath(args.model_name)
    if not os.path.isfile(model_name):
        err_msg = f'The file "{model_name}" does not exist!'
        one_peace_demo_logger.error(err_msg)
        raise ValueError(err_msg)

    sys.path.append(os.path.join(one_peace_dir))
    from one_peace.models import from_pretrained
    one_peace_demo_logger.info('ONE-PEACE is attached.')

    current_workdir = os.getcwd()
    one_peace_demo_logger.info(f'Current working directory: {current_workdir}')
    os.chdir(one_peace_dir)
    one_peace_demo_logger.info(f'New working directory: {os.getcwd()}')
    model = from_pretrained(model_name, device=device, dtype=args.torch_dtype)
    one_peace_demo_logger.info('Model is loaded.')
    os.chdir(current_workdir)
    one_peace_demo_logger.info(f'Restored working directory: {os.getcwd()}')

    src_tokens = model.process_text(['cow', 'dog', 'elephant'])
    one_peace_demo_logger.info('Texts are prepared.')

    src_images = model.process_image([
        os.path.join(one_peace_dir, 'assets', 'dog.JPEG'),
        os.path.join(one_peace_dir, 'assets', 'elephant.JPEG')
    ])
    one_peace_demo_logger.info('Images are prepared.')

    src_audios, audio_padding_masks = model.process_audio([
        os.path.join(one_peace_dir, 'assets', 'cow.flac'),
        os.path.join(one_peace_dir, 'assets', 'dog.flac')
    ])
    one_peace_demo_logger.info('Audios are prepared.')

    with torch.no_grad():
        text_features = model.extract_text_features(src_tokens)
        one_peace_demo_logger.info(f'Text embeddings are calculated (shape = {text_features.size()}).')
        image_features = model.extract_image_features(src_images)
        one_peace_demo_logger.info(f'Image embeddings are calculated (shape = {image_features.size()}).')
        audio_features = model.extract_audio_features(src_audios, audio_padding_masks)
        one_peace_demo_logger.info(f'Audio embeddings are calculated (shape = {audio_features.size()}).')

        i2t_similarity = image_features @ text_features.T
        a2t_similarity = audio_features @ text_features.T

    one_peace_demo_logger.info(f'Image-to-text similarities: {i2t_similarity.cpu().type(torch.FloatTensor).numpy()}')
    one_peace_demo_logger.info(f'Audio-to-text similarities: {a2t_similarity.cpu().type(torch.FloatTensor).numpy()}')
    del i2t_similarity, a2t_similarity

    other_src_tokens = model.process_text(
        [
            'Cattle (Bos taurus) are large, domesticated, bovid ungulates. They are prominent modern members of '
            'the subfamily Bovinae and the most widespread species of the genus Bos. '
            'Mature female cattle are referred to as cows and mature male cattle are referred to as bulls. '
            'Colloquially, young female cattle (heifers), young male cattle (bullocks), and castrated male cattle '
            '(steers) are also referred to as "cows".',
            'The dog (Canis familiaris or Canis lupus familiaris) is a domesticated descendant of the wolf. '
            'Also called the domestic dog, it is derived from extinct Pleistocene wolves, and the modern wolf is '
            'the dog\'s nearest living relative. The dog was the first species to be domesticated by humans. '
            'Hunter-gatherers did this, over 15,000 years ago, which was before the development of agriculture. '
            'Due to their long association with humans, dogs have expanded to a large number of domestic individuals '
            'and gained the ability to thrive on a starch-rich diet that would be inadequate for other canids.',
            'Elephants are the largest living land animals. Three living species are currently recognised: '
            'the African bush elephant, the African forest elephant, and the Asian elephant. They are the only '
            'surviving members of the family Elephantidae and the order Proboscidea; extinct relatives include '
            'mammoths and mastodons. Distinctive features of elephants include a long proboscis called a trunk, '
            'tusks, large ear flaps, pillar-like legs, and tough but sensitive grey skin. The trunk is prehensile, '
            'bringing food and water to the mouth and grasping objects. Tusks, which are derived from the incisor '
            'teeth, serve both as weapons and as tools for moving objects and digging. The large ear flaps assist '
            'in maintaining a constant body temperature as well as in communication. African elephants have larger '
            'ears and concave backs, whereas Asian elephants have smaller ears and convex or level backs.'
        ]
    )
    one_peace_demo_logger.info('Other texts are prepared.')

    with torch.no_grad():
        other_text_features = model.extract_text_features(other_src_tokens)
        one_peace_demo_logger.info(f'Other text embeddings are calculated (shape = {other_text_features.size()}).')
        i2t_similarity = image_features @ other_text_features.T
        a2t_similarity = audio_features @ other_text_features.T

    one_peace_demo_logger.info(f'Image-to-text similarities: {i2t_similarity.cpu().type(torch.FloatTensor).numpy()}')
    one_peace_demo_logger.info(f'Audio-to-text similarities: {a2t_similarity.cpu().type(torch.FloatTensor).numpy()}')


if __name__ == '__main__':
    one_peace_demo_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    one_peace_demo_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('one_peace_embeddings_demo.log')
    file_handler.setFormatter(formatter)
    one_peace_demo_logger.addHandler(file_handler)
    main()
