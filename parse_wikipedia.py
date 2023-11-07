from argparse import ArgumentParser
import codecs
import csv
import gc
import logging
import os
import random
import re
import sys


wikipedia_logger = logging.getLogger(__name__)
RANDOM_SEED: int = 42
MIN_WORDS_PER_PARAGRAPH: int = 5
MAX_PARAGRAPHS_PER_FILE: int = 1_000_000


def clear_text(old_text: str) -> str:
    re_for_hypertext = re.compile(r'&lt;.+?&gt;')
    return ' '.join(re_for_hypertext.sub(' ', old_text).strip().split())


def parse_wiki_file(fname: str):
    if not os.path.isfile(fname):
        err_msg = f'The file "{fname}" does not exist!'
        wikipedia_logger.error(err_msg)
        raise IOError(err_msg)
    document_is_completed = True
    document_content = []
    document_title = ''
    line_idx = 1
    with codecs.open(fname, mode='r', encoding='utf-8') as fp:
        curline = fp.readline()
        while len(curline) > 0:
            prepline = curline.strip()
            if len(prepline) > 0:
                if prepline.startswith('<doc id=') and prepline.endswith('>'):
                    if not document_is_completed:
                        err_msg = (f'The file "{fname}", line {line_idx}: '
                                   f'the document "{document_title}" is not completed!')
                        wikipedia_logger.error(err_msg)
                        raise ValueError(err_msg)
                    title_start_idx = prepline.find('title="')
                    if title_start_idx < 0:
                        err_msg = (f'The file "{fname}", line {line_idx}: '
                                   f'the document header is wrong, because the title is not found.')
                        wikipedia_logger.error(err_msg)
                        raise ValueError(err_msg)
                    if not prepline.endswith('">'):
                        err_msg = (f'The file "{fname}", line {line_idx}: '
                                   f'the document header is wrong, because the title is not found.')
                        wikipedia_logger.error(err_msg)
                        raise ValueError(err_msg)
                    title_start_idx += len('title="')
                    title_end_idx = len(prepline) - 2
                    document_title = ' '.join(prepline[title_start_idx:title_end_idx].strip().split())
                    if len(document_title) == 0:
                        err_msg = f'The file "{fname}", line {line_idx}: the document title is empty.'
                        wikipedia_logger.error(err_msg)
                        raise ValueError(err_msg)
                    document_is_completed = False
                elif prepline == '</doc>':
                    if document_is_completed:
                        err_msg = (f'The file "{fname}", line {line_idx}: unexpected completion of '
                                   f'the document "{document_title}"!')
                        wikipedia_logger.error(err_msg)
                        raise ValueError(err_msg)
                    if len(document_content) > 0:
                        yield {'title': document_title, 'paragraphs': document_content}
                    document_is_completed = True
                    document_title = ''
                    del document_content
                    document_content = []
                else:
                    words_of_paragraph = clear_text(prepline).split()
                    paragraph = ' '.join(words_of_paragraph)
                    if not((len(document_content) == 0) and (paragraph == document_title)):
                        if len(words_of_paragraph) >= MIN_WORDS_PER_PARAGRAPH:
                            document_content.append(paragraph)
                    del words_of_paragraph, paragraph
            curline = fp.readline()
            line_idx += 1
    if not document_is_completed:
        err_msg = f'The file "{fname}", line {line_idx}: unexpected completion of the document {document_title}!'
        wikipedia_logger.error(err_msg)
        raise ValueError(err_msg)


def iterate_wiki_files(wiki_dir: str):
    re_for_wiki_name = re.compile(r'^wiki_\d+')
    if not os.path.isdir(wiki_dir):
        err_msg = f'The directory "{wiki_dir}" does not exist!'
        wikipedia_logger.error(err_msg)
        raise IOError(err_msg)
    dir_items = list(map(
        lambda it2: os.path.join(wiki_dir, it2),
        filter(
            lambda it1: it1 not in {'.', '..'},
            os.listdir(wiki_dir)
        )
    ))
    if len(dir_items) > 0:
        subdirs = sorted(list(filter(lambda it: os.path.isdir(it), dir_items)))
        wikipedia_files = sorted(list(filter(
            lambda it: os.path.isfile(it) and (re_for_wiki_name.search(os.path.basename(it.lower())) is not None),
            dir_items
        )))
        for cur_fname in wikipedia_files:
            yield from parse_wiki_file(cur_fname)
        if len(subdirs) > 0:
            for cur_subdir in subdirs:
                yield from iterate_wiki_files(cur_subdir)


def main():
    random.seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_wiki_dir', type=str, required=True,
                        help='The input path to Wikipedia parsed by WikiExtractor.')
    parser.add_argument('-o', '--output', dest='output_csv', type=str, required=True,
                        help='The output name of the CSV file with structured Wikipedia content.')
    args = parser.parse_args()

    input_wiki_dir = os.path.normpath(args.input_wiki_dir)
    if not os.path.isdir(input_wiki_dir):
        err_msg = f'The directory "{input_wiki_dir}" does not exist!'
        wikipedia_logger.error(err_msg)
        raise IOError(err_msg)

    output_fname_template = os.path.normpath(args.output_csv)
    output_dir = os.path.dirname(output_fname_template)
    output_base_fname = os.path.basename(output_fname_template)
    if len(output_dir) > 0:
        if not os.path.isdir(output_dir):
            err_msg = f'The directory "{output_dir}" does not exist!'
            wikipedia_logger.error(err_msg)
            raise IOError(err_msg)
    if (not output_base_fname.lower().endswith('.csv')) or (len(output_base_fname) <= 4):
        err_msg = f'The file "{output_base_fname}" is not CSV!'
        wikipedia_logger.error(err_msg)
        raise IOError(err_msg)

    corpus = []
    counter = 1
    paragraphs_number = 0
    documents_number = 0
    for document in iterate_wiki_files(input_wiki_dir):
        for paragraph_id, paragraph_text in enumerate(document['paragraphs']):
            corpus.append((document['title'], f'{paragraph_id + 1}', paragraph_text))
            paragraphs_number += 1
        documents_number += 1
        if len(corpus) >= MAX_PARAGRAPHS_PER_FILE:
            new_output_fname = os.path.join(output_dir, output_base_fname[:-4] + '_{0:>04}.csv'.format(counter))
            with codecs.open(new_output_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
                csv_writer = csv.writer(fp, delimiter=',', quotechar='"')
                csv_writer.writerow(['title', 'paragraph_id', 'paragraph_content'])
                for title, paragraph_id, paragraph_text in corpus:
                    csv_writer.writerow([title, paragraph_id, paragraph_text])
            wikipedia_logger.info(f'{documents_number} documents and {paragraphs_number} paragraphs are processed.')
            del corpus
            corpus = []
            counter += 1
            gc.collect()
    if len(corpus) > 0:
        new_output_fname = os.path.join(output_dir, output_base_fname[:-4] + '_{0:>04}.csv'.format(counter))
        with codecs.open(new_output_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
            csv_writer = csv.writer(fp, delimiter=',', quotechar='"')
            csv_writer.writerow(['title', 'paragraph_id', 'paragraph_content'])
            for title, paragraph_id, paragraph_text in corpus:
                csv_writer.writerow([title, paragraph_id, paragraph_text])
        wikipedia_logger.info(f'{documents_number} documents and {paragraphs_number} paragraphs are processed.')
    if paragraphs_number == 0:
        err_msg = f'The Wikipedia directory "{input_wiki_dir}" is empty!'
        wikipedia_logger.error(err_msg)
        raise ValueError(err_msg)


if __name__ == '__main__':
    wikipedia_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    wikipedia_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('wikipedia_parser.log')
    file_handler.setFormatter(formatter)
    wikipedia_logger.addHandler(file_handler)
    main()
