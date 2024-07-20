import json
import sys
import os 
from generate import generate_full_prompt, setup_model_and_tokenizer


def main(dir_dataset, dir_path):    

    model, processor = setup_model_and_tokenizer()


    with open(dir_dataset, 'r') as f:
        data = json.load(f)

    for item in data:
        item['image'] = os.path.join(dir_path + item['image'])

    for item in data:
        cur_query_list = [{'text': item['conversations'][0]['value'], 'image': item['image']}]
        
        item['conversations'][0]['value'], _ = generate_full_prompt(model, cur_query_list, ('', ''))


    with open('your_file.json', 'w') as f:
        json.dump(data, f, indent=4)



if __name__ == "__main__":

    dir_path = sys.argv[2]
    dir_dataset = sys.argv[1]
    main(dir_dataset, dir_path)
