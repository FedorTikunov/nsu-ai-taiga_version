import pandas as pd
from bert_score import score
from team_code.generate import setup_model_and_tokenizer, generate_text, get_ppl

model, tokenizer = setup_model_and_tokenizer()
df = pd.read_csv('dataset_wide.csv')

answers = []
refs = []
history_list = ("", "")
for i in range(len(df)):
    refs.append(df['output'][i])
    cur = df['input'][i]
    cur_query_list = [{"type": "text", "content": cur}]
    answer, _ = generate_text(model, tokenizer, cur_query_list=cur_query_list, history_list=history_list)
    answers.append(answer)
    
P, R, F1 = score(answers, refs, lang='en', verbose=True)

print(f"BERT F1 score for this LLM: {F1.mean():.3f}")