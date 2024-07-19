from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, Message
from aiogram.filters import Command
import config
import asyncio
import requests
import os
import sys

bot = Bot(token=os.getenv('BOT_TOKEN'))
dp = Dispatcher()

@dp.message(Command(commands=['clean']))
async def on_clean(message: Message):
    ans = requests.post(f"{sys.argv[1]}/clear_context", data={"chat_id": message.chat.id})
    await message.answer(ans.text)

@dp.message(Command(commands=['set_promt']))
async def on_promt_set(message: Message):
    new_promt = " ".join(message.text.split()[1:])
    ans = requests.post(f"{sys.argv[1]}/set_param", data={"chat_id": message.chat.id, "param": "initial_promt", "value": new_promt})
    await message.answer(ans.text)

@dp.message(Command(commands=['set_history']))
async def on_history_set(message: Message):
    value = bool(int(message.text.split()[1]))
    ans = requests.post(f"{sys.argv[1]}/set_param", data={"chat_id": message.chat.id, "param": "history", "value": value})
    await message.answer(ans.text)

@dp.message(Command(commands=['set_param']))
async def on_param_set(message: Message):
    # value = bool(int(message.text.split()[1]))
    _, param, *value = message.text.split()
    value = " ".join(value)
    try:
        value = int(value)
    except:
        pass
    ans = requests.post(f"{sys.argv[1]}/set_param", data={"chat_id": message.chat.id, "param": param, "value": value})
    await message.answer(ans.text)

@dp.message(Command(commands=['get_param']))
async def on_param_get(message: Message):
    # value = bool(int(message.text.split()[1]))
    param = message.text.split()[1]
    ans = requests.post(f"{sys.argv[1]}/get_param", data={"chat_id": message.chat.id, "param": param})
    await message.answer(f"{param}: {ans.text}")

@dp.message(Command(commands=['reload']))
async def reload(message: Message):
    ans = requests.post(f"{sys.argv[1]}/reload")
    await message.answer(ans)

@dp.message()
async def on_message(message: Message):
    chat_id = message.chat.id
    
    http_form = {"chat_id": chat_id}
    form_files = {}

    if message.text:
        http_form['message'] = message.text
    elif message.caption:
        http_form['message'] = message.caption
    # file_id: (file_size)

    photos = {}
    if message.photo:
        for photo in message.photo:
            if photo.file_unique_id not in photos or photo.file_size > photos[photo.file_unique_id][1]:
                photos[photo.file_unique_id] = (photo.file_id, photo.file_size)
    
    for file_unique_id, (file_id, _) in photos.items():
        file = await bot.get_file(file_id)
        file = await bot.download_file(file.file_path)
        form_files[file_unique_id] = (photo.file_unique_id, file, 'image/jpeg')
    
    if message.document is not None:
        doc = message.document
        file = await bot.get_file(doc.file_id)
        file = await bot.download_file(file.file_path)
        form_files[doc.file_unique_id] = (doc.file_name, file, doc.mime_type)
    
    resp = requests.post(f"{sys.argv[1]}/send", data=http_form, files=form_files)

    await message.answer(text=resp.text)
    # await message.answer(text=resp.text)

if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot))
