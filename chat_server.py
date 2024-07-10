import flask
from team_code.generate import setup_model_and_tokenizer, generate_text
import logging
from flask import request

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

app = flask.Flask(__name__)
app.secret_key = 'BAD_SECRET_KEY'

model, tokenizer = setup_model_and_tokenizer()

global_history = {}

@app.route('/')
def index():
    """
    main page for chat
    """
    return """
<html>
<head>
    <title>Chat</title>
    <meta charset="utf-8" />
    <script>

        function send() {
            fetch('/send', {
                method: 'POST',
                body: document.getElementsByName('message')[0].value
            }).then(function (response) {
                return response.text();
            }).then(function (text) {
                document.getElementById('answer').value = text;
            });
        }

    </script>
</head>
<body>
    <h1>Chat</h1>
    <form action="javascript:send()" method="post">
        <input type="text" name="message" />
        <input type="submit" value="Send" />
    </form>
    <textarea id="answer" cols="80" rows="20"></textarea>
</body>
</html>
"""

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

@app.route('/send', methods=['POST'])
def send():
    global global_history

    cur_chat_id = "chat1"
    if cur_chat_id in global_history:
        history_list = global_history[cur_chat_id]
    else:
        history_list = ("", "")
        global_history[cur_chat_id] = history_list

    logging.info(f"Received text message from chat_id {cur_chat_id}")

    message = request.body

    try:
        if message.text == "Контекст очищен" or message.text == "Generating answer...":
            return
        elif message.text == '/clear_context':
            global_history[cur_chat_id] = ("", "")
            logging.info(f"Clear context for chat_id {cur_chat_id}")
            return "Контекст очищен"
        elif message.text == '/help':
            # await message.answer(text=help_message)
            logging.info(f"Show help message for chat_id {cur_chat_id}")
            return help_message
        else:
            cur_query_list = []

            # tmp_message = await message.answer(text="Generating answer...")
            # loading_gif = open("resources/loading.gif", 'rb')
            logging.info(f"Show loading.gif for chat_id {cur_chat_id}")

            cur_query_list.append({'type': 'text', 'content': message.text})

            answer, new_history_list = generate_text(model,
                                                    tokenizer,
                                                    cur_query_list=cur_query_list,
                                                    history_list=history_list)
            # answer, new_history_list = "test", history_list
            # await asyncio.sleep(3)

            global_history[cur_chat_id] = new_history_list

            logging.info(f"Send text answer for chat_id {cur_chat_id}")
            return answer

    except Exception as e:
        logging.error(e, exc_info=True)
        return e

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
