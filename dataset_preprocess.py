import json
import sys
import os 
from pathlib import Path
from team_code.generate import generate_full_prompt, setup_model_and_tokenizer
import tqdm


def main(dir_dataset: str, dir_path: str, save_path: str):
    dir_dataset: Path = Path(dir_dataset)
    dir_path: Path = Path(dir_path)
    save_path: Path = Path(save_path)

    model, processor = setup_model_and_tokenizer()


    with open(dir_dataset, 'r') as f:
        data = json.load(f)

        new_data = []

    for item in tqdm.tqdm(data):
        img_full_path: Path = dir_path / item['image']
        if not img_full_path.exists():
            continue
        item['image'] = str(dir_path / item['image'])
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
        
        item['conversations'][0]['value'] = generate_full_prompt(model, cur_query_list, ('', ''))
        new_data.append(item)


    with open(save_path, 'w') as f:
        json.dump(new_data, f, indent=4)



if __name__ == "__main__":

    dir_path = sys.argv[2]
    dir_dataset = sys.argv[1]
    save_path = sys.argv[3]
    main(dir_dataset, dir_path, save_path)