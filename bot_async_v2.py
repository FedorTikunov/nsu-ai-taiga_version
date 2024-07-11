from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, Message
import config
import asyncio
import config

if config.debug:
    from debug.testing import setup_model_and_tokenizer, generate_text
else:
    from team_code.generate import setup_model_and_tokenizer, generate_text

bot = Bot(token=config.token)
dp = Dispatcher()

model, tokenizer = setup_model_and_tokenizer()

global_history = {}

@dp.message()
async def on_message(message: Message):
    chat_id = message.chat.id
    if chat_id not in global_history:
        global_history[chat_id] = ("", "")
    
    cur_query_list = []
    if message.text:
        cur_query_list.append({'type': 'text', 'content': message.text})

    if cur_query_list:
        history_list = global_history[chat_id]
        answer, new_history_list = generate_text(model,
                                                tokenizer,
                                                cur_query_list=cur_query_list,
                                                history_list=history_list)
        await message.answer(text=answer)
    else:
        await message.answer(text="Ничего не написано")

if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot))
