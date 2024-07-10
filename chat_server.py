import flask
import logging
from flask import request
import config
from pathlib import Path


if config.debug:
    from debug.testing import setup_model_and_tokenizer, generate_text
else:
    from team_code.generate import setup_model_and_tokenizer, generate_text

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

app = flask.Flask(__name__)
app.secret_key = 'BAD_SECRET_KEY'

model, tokenizer = setup_model_and_tokenizer()

global_history = {}

_SCRIPT_DIR = Path(__file__).parent
PHOTO_DIR = _SCRIPT_DIR / "photo"
if not PHOTO_DIR.exists():
    PHOTO_DIR.mkdir()
VOISE_DIR = _SCRIPT_DIR / "voice"
if not VOISE_DIR.exists():
    VOISE_DIR.mkdir()

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
            let form_data = new FormData();
            let file = document.getElementById('file');
            form_data.append('file', file.files[0]);
            form_data.append('message', document.getElementsByName('message')[0].value);
            console.log(form_data);
            fetch('/send', {
                method: 'POST',
                body: form_data
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
        <textarea name="message" cols="80"></textarea><br>
        <input type="file" id="file" name="file"/><br>
        <input type="submit" value="Send" /><br>
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
    
    cur_query_list = []

    if "file" in request.files:
        file = request.files['file']
        file_type = file.content_type.split("/")[0]
        file_path = PHOTO_DIR / file.filename
        file.save(file_path)

        # BUGMAYBE: file_type not the same as js content_type
        cur_query_list.append({'type': file_type, 'content': file_path.absolute()})

    message = request.form['message']
    if message:
        cur_query_list.append({'type': 'text', 'content': message})
    
    # elif message == '/clear_context':
    #     global_history[cur_chat_id] = ("", "")
    #     logging.info(f"Clear context for chat_id {cur_chat_id}")
    #     return "Контекст очищен"
    if cur_query_list:
        try:
            answer, new_history_list = generate_text(model,
                                                    tokenizer,
                                                    cur_query_list=cur_query_list,
                                                    history_list=history_list)

            global_history[cur_chat_id] = new_history_list

            return answer
        except Exception as e:
            logging.error(e, exc_info=True)
            return str(e)
    else:
        return "В запросе ничего нет"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=config.flask_debug)
    # app.run(host='localhost', port=5000)
