from argparse import ArgumentParser
import codecs
import copy
import csv
import gc
import logging
import os
import random
import re
import sys
from typing import List, Tuple


vectorization_logger = logging.getLogger(__name__)


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
    pass


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
