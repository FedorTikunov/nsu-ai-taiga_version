from typing import Tuple, List, Dict
import torch.nn as nn
import torch
import logging


def prepare_logger():
    conversation_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    conversation_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('multimodal_conversation.log')
    file_handler.setFormatter(formatter)
    conversation_logger.addHandler(file_handler)