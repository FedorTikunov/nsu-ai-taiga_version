from team_code.generate import setup_model_and_tokenizer, generate_text, get_ppl
from tqdm import tqdm
import nltk
nltk.download('punkt')
import telebot
from telebot import types
import math
import os
import uuid
import config.runtime_config as runtime_config


def make_dir(dir_name):
    isExist = os.path.exists(dir_name)
    if not isExist:
        os.mkdir(dir_name)


make_dir('audio')
make_dir('ready')
make_dir('voice')
make_dir('photo')


model, tokenizer = setup_model_and_tokenizer()
bot = telebot.TeleBot(runtime_config.token)

history_list = ("", "")


@bot.message_handler(commands=['start'])
def send_welcome(message):

    global history_list
    history_list = ("", "")
#     hideBoard = types.ReplyKeyboardRemove()
    msg = bot.send_message(message.chat.id, text ='Здравствуй, {0.first_name}! Я мультимодальный диалоговый ассистент NSU AI, чем могу помочь?'.format(message.from_user))


        
@bot.message_handler(content_types=['text'])
def handle_text(message):
    markup=types.ReplyKeyboardMarkup(resize_keyboard=True)
    global history_list

    try:
        
        if message.text =="Контекст очищен" or message.text == "Generating answer...":
            return
        if message.text == 'Очистить контекст':
            history_list = ("","")
            bot.send_message(message.chat.id, text="Контекст очищен")
        else:
            cur_query_list = []

            msg = bot.send_message(message.chat.id, text="Generating answer...")
            bot.last_message_sent = msg.chat.id, msg.message_id

            cur_query_list.append({'type': 'text', 'content': message.text})
            btn1 = types.KeyboardButton("Очистить контекст")

            markup.add(btn1)
            answer, new_history_list = generate_text(model, tokenizer, cur_query_list=cur_query_list, history_list=history_list)
            history_list = new_history_list
            bot.delete_message(*bot.last_message_sent)
            bot.send_message(message.chat.id, text=answer, reply_markup=markup)
        
        


    except Exception:
        bot.send_message(message.chat.id, text="Попробуйте написать текст запроса снова", reply_markup=markup)


    
@bot.message_handler(content_types=['photo'])
def handle_image(message):
    markup=types.ReplyKeyboardMarkup(resize_keyboard=True)
            
    btn1 = types.KeyboardButton("Очистить контекст")

    markup.add(btn1)
    cur_query_list = []


    try:
        msg = bot.send_message(message.chat.id, text="Generating answer...")
        if message.caption is not None:
            cur_query_list.append({'type': 'text', 'content': message.caption})
        else:
            cur_query_list.append({'type': 'text', 'content': 'Describe what you see in this picture?'})
        bot.last_message_sent = msg.chat.id, msg.message_id
        filename = str(uuid.uuid4())
        file_name_full="./photo/"+filename +'.jpg'
        fileID = message.photo[-1].file_id
        file_info = bot.get_file(fileID)
        photo = bot.download_file(file_info.file_path)
        
        with open(file_name_full, 'wb') as new_file:
            new_file.write(photo)
        cur_query_list.append({'type': 'image', 'content': file_name_full})

        global history_list
        answer, new_history_list = generate_text(model, tokenizer, cur_query_list=cur_query_list, history_list=history_list)
        history_list = new_history_list
        bot.delete_message(*bot.last_message_sent)

        bot.send_message(message.chat.id, text=answer, reply_markup=markup)

       
        
    except Exception:
        
        bot.send_message(message.chat.id, text="Возникла проблема с вашим фото, попробуйте ещё раз", reply_markup=markup)

        
        
@bot.message_handler(content_types=['audio', 'document'])
def handle_audio(message):
    markup=types.ReplyKeyboardMarkup(resize_keyboard=True)
        
    btn1 = types.KeyboardButton("Очистить контекст")

    markup.add(btn1)    

    try:
        msg = bot.send_message(message.chat.id, text="Generating answer...")
        bot.last_message_sent = msg.chat.id, msg.message_id
        cur_query_list = []
        if message.caption is not None:
            cur_query_list.append({'type': 'text', 'content': message.caption})
        else:
            cur_query_list.append({'type': 'text', 'content': 'Describe what you hear in this audio?'})
        filename = str(uuid.uuid4())
        file_name_full="./audio/"+filename
        file_name_full_converted="./ready/"+filename+".wav"
        if message.content_type == 'audio':
            file_info = bot.get_file(message.audio.file_id)
        else:
            file_info = bot.get_file(message.document.file_id)

        downloaded_file = bot.download_file(file_info.file_path)
        with open(file_name_full, 'wb') as new_file:
            new_file.write(downloaded_file)
        os.system("ffmpeg -hide_banner -loglevel error -i "+file_name_full+"  "+file_name_full_converted)
        cur_query_list.append({'type': 'audio', 'content': file_name_full_converted})
        global history_list
        answer, new_history_list = generate_text(model, tokenizer, cur_query_list=cur_query_list, history_list=history_list)
        history_list = new_history_list
        bot.delete_message(*bot.last_message_sent)
        bot.send_message(message.chat.id, text=answer, reply_markup=markup)
    except Exception:
        bot.send_message(message.chat.id, text="Возникла проблема с вашим файлом, убедитесь, что это аудио", reply_markup=markup)


        
    
    
@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    markup=types.ReplyKeyboardMarkup(resize_keyboard=True)
        
    btn1 = types.KeyboardButton("Очистить контекст")

    markup.add(btn1)

    try:
        msg = bot.send_message(message.chat.id, text="Generating answer...")
        bot.last_message_sent = msg.chat.id, msg.message_id
        cur_query_list = []

        filename = str(uuid.uuid4())
        file_name_full="./voice/"+filename+".ogg"
        file_name_full_converted="./ready/"+filename+".wav"
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with open(file_name_full, 'wb') as new_file:
            new_file.write(downloaded_file)
        os.system("ffmpeg -hide_banner -loglevel error -i "+file_name_full+"  "+file_name_full_converted)
        
        cur_query_list.append({'type': 'text', 'content': 'Describe this voice message?'})
        
        cur_query_list.append({'type': 'audio', 'content': file_name_full_converted})

        global history_list
        answer, new_history_list = generate_text(model, tokenizer, cur_query_list=cur_query_list, history_list=history_list)
        history_list = new_history_list
        bot.delete_message(*bot.last_message_sent)
        bot.send_message(message.chat.id, text=answer, reply_markup=markup)
    except Exception:
        bot.send_message(message.chat.id, text="Возникла проблема с вашим голосовым, попробуйте ещё раз", reply_markup=markup)
        

    
bot.polling(none_stop=True)