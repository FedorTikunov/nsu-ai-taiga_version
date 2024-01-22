from team_code.generate import setup_model_and_tokenizer, generate_text, prepare_logger
import nltk
import ffmpeg

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

cur_query_list = [{'type': 'text', 'content': 'What do you see on this picture?'}, {'type': 'image', 'content': '/userspace/dra/nsu-ai/test.jpg'}]
history_list = ("", "")
answer, new_history_list = generate_text(model, tokenizer, cur_query_list=cur_query_list, history_list=history_list)
print(f'History: {new_history_list}')
print(f'Answer: {answer}')

cur_query_list = [{'type': 'text', 'content': 'What is happening?'}, {'type': 'audio', 'content': '/userspace/dra/nsu-ai/test.wav'}]
answer, new_history_list = generate_text(model, tokenizer, cur_query_list=cur_query_list, history_list=new_history_list)
print(f'History: {new_history_list}')
print(f'Answer: {answer}')
