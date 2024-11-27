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
    
    def set_dump_image(self, image):
        pass

    @torch.no_grad()
    def generate(self, msgs: List[OneMsg], dataset=None):
        return self.generate_inner(msgs, dataset)

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=1282):
        output = [self.generate(image, question, max_new_tokens) for image, question in zip(image_list, question_list)]
        return output
    
    def generate_inner(self, msgs: List[OneMsg], dataset=None):
        print(msgs)
        for msg in msgs:
            msg["content"] = msg["value"]
            del msg["value"]
        with torch.amp.autocast("cuda"):
            response, history = generate.generate_text(self.model, self.tokenizer, cur_query_list=msgs, history_list=("", ""))
        return response
