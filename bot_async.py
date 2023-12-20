import os
import uuid
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, Message
from aiogram.utils import executor
import ffmpeg  # Assuming ffmpeg-python is installed
from team_code.generate import setup_model_and_tokenizer, generate_text
import config
import logging
from aiolimiter import AsyncLimiter

limiter = AsyncLimiter(25, 1)  # 25 сообщений в секунду

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Setup model and tokenizer
model, tokenizer = setup_model_and_tokenizer()
# model, tokenizer = None, None

# Bot initialization
bot = Bot(token=config.token)
dp = Dispatcher(bot)

global_history = {}

help_message = """Ты правда не разобрался в двух кнопках? Ну ладно, распишу по пунктам:

1. Можно писать текст
2. Можно не писать текст
3. Можно отправлять изображения
4. Можно не отправлять...
...
42. Можно очистить контекст (/clear_context)
43. Можно не очищать

Удачи, кожаный!
"""


# Helper functions to create directories
def make_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)


make_dir('audio')
make_dir('ready')
make_dir('voice')
make_dir('photo')


async def setup_bot_commands(*args):
    bot_commands = [
        BotCommand(command="/help", description="Get info about me"),
        BotCommand(command="/clear_context", description="Clear current dialogue context"),
    ]
    await bot.set_my_commands(bot_commands)


# Handlers
@dp.message_handler(commands=['start'])
async def send_welcome(message: Message):
    async with limiter:
        await message.answer(f'Здравствуй, {message.from_user.first_name}! Я мультимодальный диалоговый ассистент NSU AI, чем могу помочь?')


@dp.message_handler(content_types=['text'])
async def handle_text(message: Message):
    global global_history

    cur_chat_id = message["chat"]["id"]
    if cur_chat_id in global_history:
        history_list = global_history[cur_chat_id]
    else:
        history_list = ("", "")
        global_history[cur_chat_id] = history_list

    logging.info(f"Received text message from chat_id {cur_chat_id}")

    try:
        if message.text == "Контекст очищен" or message.text == "Generating answer...":
            return
        elif message.text == '/clear_context':
            global_history[cur_chat_id] = ("", "")
            async with limiter:
                await message.answer(text="Контекст очищен")
            logging.info(f"Clear context for chat_id {cur_chat_id}")
        elif message.text == '/help':
            await message.answer(text=help_message)
            logging.info(f"Show help message for chat_id {cur_chat_id}")
        else:
            cur_query_list = []

            # tmp_message = await message.answer(text="Generating answer...")
            loading_gif = open("resources/loading.gif", 'rb')
            async with limiter:
                tmp_message = await message.answer_animation(loading_gif)
            logging.info(f"Show loading.gif for chat_id {cur_chat_id}")

            cur_query_list.append({'type': 'text', 'content': message.text})

            answer, new_history_list = generate_text(model,
                                                     tokenizer,
                                                     cur_query_list=cur_query_list,
                                                     history_list=history_list)
            # answer, new_history_list = "test", history_list
            # await asyncio.sleep(3)

            global_history[cur_chat_id] = new_history_list

            async with limiter:
                await tmp_message.delete()
                await message.answer(text=answer)
            logging.info(f"Send text answer for chat_id {cur_chat_id}")

    except Exception as e:
        logging.error(e, exc_info=True)
        async with limiter:
            await message.answer("Попробуйте написать текст запроса снова")


# Additional handlers (for photo, audio, voice) go here
@dp.message_handler(content_types=['photo'])
async def handle_image(message: Message):
    cur_query_list = []

    cur_chat_id = message["chat"]["id"]
    if cur_chat_id in global_history:
        history_list = global_history[cur_chat_id]
    else:
        history_list = ("", "")
        global_history[cur_chat_id] = history_list

    try:
        # tmp_message = await message.answer(text="Generating answer...")
        loading_gif = open("resources/loading.gif", 'rb')
        async with limiter:
            tmp_message = await message.answer_animation(loading_gif)

        if message.caption is not None:
            cur_query_list.append({'type': 'text', 'content': message.caption})
        else:
            cur_query_list.append({'type': 'text', 'content': 'Describe what you see in this picture?'})

        file_name_full = f"./photo/{str(uuid.uuid4())}.jpg"
        file_id = message.photo[-1].file_id

        async with limiter:
            file_info = await bot.get_file(file_id)
            photo = await bot.download_file(file_info.file_path)

        with open(file_name_full, 'wb') as new_file:
            new_file.write(photo.read())

        cur_query_list.append({'type': 'image', 'content': file_name_full})

        answer, new_history_list = generate_text(model,
                                                 tokenizer,
                                                 cur_query_list=cur_query_list,
                                                 history_list=history_list)
        # answer, new_history_list = "test_image", history_list
        # await asyncio.sleep(3)

        global_history[cur_chat_id] = new_history_list

        async with limiter:
            await tmp_message.delete()
            await message.answer(text=answer)

    except Exception as e:
        logging.error(e, exc_info=True)
        async with limiter:
            await message.answer("Возникла проблема с вашим фото, попробуйте ещё раз")


@dp.message_handler(content_types=['audio', 'document'])
async def handle_audio(message):
    cur_query_list = []

    cur_chat_id = message["chat"]["id"]
    if cur_chat_id in global_history:
        history_list = global_history[cur_chat_id]
    else:
        history_list = ("", "")
        global_history[cur_chat_id] = history_list

    try:
        # tmp_message = await message.answer(text="Generating answer...")
        loading_gif = open("resources/loading.gif", 'rb')
        async with limiter:
            tmp_message = await message.answer_animation(loading_gif)

        if message.caption is not None:
            cur_query_list.append({'type': 'text', 'content': message.caption})
        else:
            cur_query_list.append({'type': 'text', 'content': 'Describe what you hear in this audio?'})

        filename = str(uuid.uuid4())
        file_name_full = f"./audio/{filename}.jpg"
        file_name_full_converted = f"./ready/{filename}.wav"

        async with limiter:
            if message.content_type == 'audio':
                file_info = await bot.get_file(message.audio.file_id)
            else:
                file_info = await bot.get_file(message.document.file_id)

            downloaded_file = await bot.download_file(file_info.file_path)

        with open(file_name_full, 'wb') as new_file:
            new_file.write(downloaded_file.read())

        os.system("ffmpeg -hide_banner -loglevel error -i "+file_name_full+"  "+file_name_full_converted)
        cur_query_list.append({'type': 'audio', 'content': file_name_full_converted})

        answer, new_history_list = generate_text(model,
                                                 tokenizer,
                                                 cur_query_list=cur_query_list,
                                                 history_list=history_list)
        # answer, new_history_list = "test_audio", history_list
        # await asyncio.sleep(3)

        global_history[cur_chat_id] = new_history_list

        async with limiter:
            await tmp_message.delete()
            await message.answer(text=answer)

    except Exception as e:
        logging.error(e, exc_info=True)
        async with limiter:
            await message.answer("Возникла проблема с вашим файлом, убедитесь, что это аудио")


@dp.message_handler(content_types=['voice']) 
async def handle_voice(message):
    cur_query_list = []

    cur_chat_id = message["chat"]["id"]
    if cur_chat_id in global_history:
        history_list = global_history[cur_chat_id]
    else:
        history_list = ("", "")
        global_history[cur_chat_id] = history_list

    try:
        # tmp_message = await message.answer(text="Generating answer...")
        loading_gif = open("resources/loading.gif", 'rb')
        async with limiter:
            tmp_message = await message.answer_animation(loading_gif)

        filename = str(uuid.uuid4())
        file_name_full = f"./voice/{filename}.ogg"
        file_name_full_converted = f"./ready/{filename}.wav"

        async with limiter:
            file_info = await bot.get_file(message.voice.file_id)
            downloaded_file = await bot.download_file(file_info.file_path)

        with open(file_name_full, 'wb') as new_file:
            new_file.write(downloaded_file.read())
        os.system("ffmpeg -hide_banner -loglevel error -i "+file_name_full+"  "+file_name_full_converted)
        
        cur_query_list.append({'type': 'text', 'content': 'Describe this voice message?'})
        cur_query_list.append({'type': 'audio', 'content': file_name_full_converted})

        answer, new_history_list = generate_text(model,
                                                 tokenizer,
                                                 cur_query_list=cur_query_list,
                                                 history_list=history_list)
        # answer, new_history_list = "test_speech", history_list
        # await asyncio.sleep(3)

        global_history[cur_chat_id] = new_history_list

        async with limiter:
            await tmp_message.delete()
            await message.answer(text=answer)

    except Exception as e:
        logging.error(e, exc_info=True)
        async with limiter:
            await message.answer("Возникла проблема с вашим голосовым, попробуйте ещё раз")
        
        
# Start polling
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True, on_startup=setup_bot_commands)
