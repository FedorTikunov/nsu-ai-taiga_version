from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, Message
import config
import asyncio
import requests
import os
import sys

bot = Bot(token=os.getenv('BOT_TOKEN'))
dp = Dispatcher()

@dp.message()
async def on_message(message: Message):
    chat_id = message.chat.id
    
    http_form = {"chat_id": chat_id}
    form_files = {}

    if message.text:
        http_form['message'] = message.text
    # file_id: (file_size)

    print(message.audio)
    print(message.photo)
    print(message.document)
    print("---")

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
    
    resp = requests.post(sys.argv[1], data=http_form, files=form_files)

    await message.answer(text=resp.text)
    # await message.answer(text=resp.text)

if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot))
