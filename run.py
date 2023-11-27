from argparse import ArgumentParser
import codecs
import os
import json
from typing import Tuple

from team_code.generate import setup_model_and_tokenizer, generate_text, prepare_logger
from team_code.generate import conversation_logger


def load_previous_dialogue(fname: str) -> Tuple[str, str]:
    err_msg = f'The file "{fname}" contains a wrong dialogue history!'
    with codecs.open(fname, mode='r', encoding='utf-8') as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise IOError(err_msg + f' Expected {type({"a": 1, "b": 2})}, got {type(data)}.')
    if 'dialogue' not in data:
        raise IOError(err_msg + ' The "dialogue" key is not found.')
    if 'last_answer' not in data:
        raise IOError(err_msg + ' The "last_answer" key is not found.')
    dialogue = data['dialogue']
    last_answer = data['last_answer']
    if not isinstance(dialogue, str):
        raise IOError(err_msg + f' The "dialogue" value is incorrect. Expected {type("123")}, got {type(dialogue)}.')
    if not isinstance(last_answer, str):
        err_msg += f' The "last_answer" value is incorrect. Expected {type("123")}, got {type(last_answer)}.'
        raise IOError(err_msg)
    return dialogue, last_answer


def main():
    parser = ArgumentParser()
    parser.add_argument('-j', '--json', dest='dialogue_json', type=str, required=True,
                        help='The JSON file with previous dialogue (if it does not exist, '
                             'then the dialogue will be started from scratch).')
    parser.add_argument('-t', '--text', dest='text_question', type=str, required=True,
                        help='The user\'s text question.')
    parser.add_argument('-i', '--image', dest='image_for_question', type=str, required=False, default=None,
                        help='The user\'s supplementary image file.')
    parser.add_argument('-a', '--audio', dest='audio_for_question', type=str, required=False, default=None,
                        help='The user\'s supplementary audio file.')
    args = parser.parse_args()

    prepare_logger()

    json_fname = os.path.normpath(args.dialogue_json)
    if os.path.isfile(json_fname):
        try:
            history = load_previous_dialogue(json_fname)
        except BaseException as err:
            conversation_logger.error(f'The dialogue history cannot be loaded from the "{json_fname}". {str(err)}')
            raise
    else:
        history = ('', '')

    question = args.text_question.strip()
    if len(question) == 0:
        err_msg = 'The user\'s text question is empty.'
        conversation_logger.error(err_msg)
        raise ValueError(err_msg)
    question = ' '.join(question.split())
    content = [{'type': 'text', 'content': question}]

    if args.image_for_question is not None:
        image_fname = os.path.normpath(args.image_for_question)
        if not os.path.isfile(image_fname):
            err_msg = f'The image "{image_fname}" does not exist!'
            conversation_logger.error(err_msg)
            raise ValueError(err_msg)
        content.append([{'type': 'image', 'content': image_fname}])

    if args.audio_for_question is not None:
        audio_fname = os.path.normpath(args.audio_for_question)
        if not os.path.isfile(audio_fname):
            err_msg = f'The audio "{audio_fname}" does not exist!'
            conversation_logger.error(err_msg)
            raise ValueError(err_msg)
        content.append([{'type': 'audio', 'content': audio_fname}])

    model, tokenizer = setup_model_and_tokenizer()
    answer, new_history = generate_text(model, tokenizer, cur_query_list=content, history_list=history)
    conversation_logger.info(f'Answer: {answer}')
    with codecs.open(json_fname, mode='w', encoding='utf-8') as fp:
        json.dump(
            obj={
                'dialogue': new_history[0],
                'last_answer': new_history[1]
            },
            fp=fp,
            ensure_ascii=False,
            indent=4
        )


if __name__ == "__main__":
    main()
