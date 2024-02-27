import json
from typing import List, Dict

import pandas as pd
from bert_score import score

from team_code.generate import setup_model_and_tokenizer, generate_text

model, tokenizer = setup_model_and_tokenizer()
system_prompts = pd.read_csv('system_prompts.csv')
dataset_path = "Meno-Data-Set.jsonl"
dialogues = []
try:
    with open(dataset_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                dialogue = json.loads(line)
                dialogues.append(dialogue)
            except json.JSONDecodeError as e:
                print(f'Error during parsing JSON: {e}')
except FileNotFoundError:
    print(f'File "{dataset_path}" not found.')

prompts_and_scores = []
histories = []

for system_prompt in system_prompts.values:
    F1s_for_prompt = []
    Ps_for_prompt = []
    Rs_for_prompt = []
    refs = []
    answers = []
    for dialogue in dialogues:
        history_list = ("", "")
        cur_query_list: List[Dict[str, str]] = []
        for i in range(len(dialogue)):
            if i % 2 == 0:
                cur_query_list = dialogue[i]
            if i % 2 == 1:
                answer, _ = generate_text(model, tokenizer, cur_query_list=cur_query_list, history_list=history_list,
                                          system_prompt=system_prompt[0])
                best_ref = ""
                max_f1_score = 0.0
                if len(dialogue[i]) != 1:
                    for ref_from_dataset in dialogue[i]:
                        _, _, F1_options = score([answer], [ref_from_dataset], lang='en', verbose=True)
                        if F1_options.item() > max_f1_score:
                            max_f1_score = F1_options.item()
                            best_ref = ref_from_dataset
                    refs.append(best_ref)
                else:
                    refs.append(dialogue[i][0])
                answers.append(answer)
        histories.append(history_list[0] + history_list[1])
    P_for_dialogues, R_for_dialogues, F1_for_dialogues = score(answers, refs, lang='en', verbose=True)
    F1_for_prompt = F1_for_dialogues.mean().item()
    P_for_prompt = P_for_dialogues.mean().item()
    R_for_prompt = R_for_dialogues.mean().item()
    prompts_and_scores.append(
        {"prompt": system_prompt.item(),
         "F1_score": F1_for_prompt,
         "P_score": P_for_prompt,
         "R_score": R_for_prompt}
    )

with open('prompts_and_scores.json', 'w') as f:
    json.dump(prompts_and_scores, f, indent=4)

with open('histories.json', 'w') as f:
    json.dump(histories, f, indent=4)
