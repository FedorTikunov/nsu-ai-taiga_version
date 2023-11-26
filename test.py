from team_code.generate import setup_model_and_tokenizer, generate_text, prepare_logger


prepare_logger()
model, tokenizer = setup_model_and_tokenizer()

# test 1
cur_query_list = [{'type': 'text', 'content': 'What is the smallest country in the world?'}]
history_list = ("", "")
answer, new_history_list = generate_text(model, tokenizer, cur_query_list=cur_query_list, history_list=history_list)
print(f'History: {new_history_list}')
print(f'Answer: {answer}')

cur_query_list = [{'type': 'text', 'content': 'Where is it located?'}]
answer, new_history_list = generate_text(model, tokenizer, cur_query_list=cur_query_list, history_list=new_history_list)
print(f'History: {new_history_list}')
print(f'Answer: {answer}')

messages = [
    {"role": "user", "content": "What is the smallest country in the world?"},
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds

generated_ids = model.llm.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
