import json
import sys
import os 
from pathlib import Path
from generate import generate_full_prompt, setup_model_and_tokenizer


def main(dir_dataset: str, dir_path: str):
    dir_dataset: Path = Path(dir_dataset)
    dir_path: Path = Path(dir_path)

    model, processor = setup_model_and_tokenizer()


    with open(dir_dataset, 'r') as f:
        data = json.load(f)

    for item in data:
        item['image'] = str(dir_path / item['image'])

    for item in data:
        cur_query_list = [
            {
                "type": "image",
                "content": item["image"],
            },
            {
                "type": "text",
                "content": item['conversations'][0]['value'],
            }
            ]
        
        item['conversations'][0]['value'], _ = generate_full_prompt(model, cur_query_list, ('', ''))


    with open(dir_dataset, 'w') as f:
        json.dump(data, f, indent=4)



if __name__ == "__main__":

    dir_path = sys.argv[2]
    dir_dataset = sys.argv[1]
    main(dir_dataset, dir_path)
