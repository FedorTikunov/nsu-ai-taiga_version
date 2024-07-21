import flask
import logging
from flask import request
import config.runtime_config as runtime_config
import config.startup_config as startup_config
from pathlib import Path
import importlib


if startup_config.debug:
    import debug.testing as generate
else:
    import team_code.generate as generate

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

app = flask.Flask(__name__)
app.secret_key = 'BAD_SECRET_KEY'

model, tokenizer = generate.setup_model_and_tokenizer()

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
        <input type="hidden" name="chat_id" value="chat1" />
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

    if "chat_id" in request.form:
        cur_chat_id = request.form['chat_id']
    else:
        cur_chat_id = "chat1"

    if cur_chat_id in global_history:
        history_list = global_history[cur_chat_id]
    else:
        history_list = ("", "")
        global_history[cur_chat_id] = history_list

    logging.info(f"Received text message from chat_id {cur_chat_id}")
    
    cur_query_list = []

    for key, f_obj in request.files.items():
        file_type = f_obj.content_type.split("/")[0]
        file_path = PHOTO_DIR / f_obj.filename
        f_obj.save(file_path)

        # BUGMAYBE: file_type not the same as js content_type
        cur_query_list.append({'type': file_type, 'content': file_path.absolute()})

    if "message" in request.form and request.form['message']:
        cur_query_list.append({'type': 'text', 'content': request.form['message']})
    
    # elif message == '/clear_context':
    #     global_history[cur_chat_id] = ("", "")
    #     logging.info(f"Clear context for chat_id {cur_chat_id}")
    #     return "Контекст очищен"
    if cur_query_list:
        try:
            answer, new_history_list = generate.generate_text(model,
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

@app.route("/clear_context", methods=['POST'])
def clear_context():
    global global_history
    if "chat_id" in request.form:
        cur_chat_id = request.form['chat_id']
    else:
        cur_chat_id = "chat1"
    global_history[cur_chat_id] = ("", "")
    return "Контекст очищен"

@app.route("/set_param", methods=['POST'])
def set_param():
    if "param" in request.form and "value" in request.form:
        value = request.form['value']
        try:
            value = int(value)
        except:
            pass
        runtime_config.__dict__[request.form['param']] = value
        return f"Значение {request.form['param']} установлено в {value}"
    return "Промт не установлен"

@app.route("/get_param", methods=['POST'])
def get_param():
    if "param" in request.form and request.form["param"] in runtime_config.__dict__:
        param = runtime_config.__dict__[request.form['param']]
        return f"{type(param)} = '{param}'"
    return "Неизвестный параметр"

@app.route("/reload", methods=['POST'])
def reload_module():
    importlib.reload(generate)
    return "Модуль перезагружен"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=startup_config.flask_debug)
    # app.run(host='localhost', port=5000)
