import team_code.generate as generate
import torch
from pathlib import Path
from typing import List, Dict, TypedDict, Union, Literal


class OneMsg(TypedDict):
    type: Union[Literal["text"], Literal["image"]]
    value: Union[str, Union[Path, str]]


class OnlyFansModel:

    def __init__(self):
        self.model, self.tokenizer = generate.setup_model_and_tokenizer()

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=64):
        if isinstance(image, Path):
            image = str(image.absolute())
        assert isinstance(image, str) 
        with torch.cuda.amp.autocast():
            cur_query_list = [{'type': 'image', 'content': image}, {'type': 'text', 'content': question}]
            response, history = generate.generate_text(self.model, self.tokenizer, cur_query_list=cur_query_list, history_list=("", ""))
        return response

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=1282):
        output = [self.generate(image, question, max_new_tokens) for image, question in zip(image_list, question_list)]
        return output
    
    def generate_inner(self, msgs: List[OneMsg], dataset=None):
        for msg in msgs:
            msg["content"] = msg["value"]
            del msg["value"]
        with torch.cuda.amp.autocast():
            response, history = generate.generate_text(self.model, self.tokenizer, cur_query_list=msgs, history_list=("", ""))
        return response
