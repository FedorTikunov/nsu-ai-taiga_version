from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, Message
import config
import asyncio
import config
import requests

bot = Bot(token=config.token)
dp = Dispatcher()

@dp.message()
async def on_message(message: Message):
    chat_id = message.chat.id
    
    http_form = {"chat_id": chat_id}

    if message.text:
        http_form['message'] = message.text
    
    resp = requests.post("http://localhost:8888/send", data=http_form)

    await message.answer(text=resp.text)

if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot))
