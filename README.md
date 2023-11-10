# NSU-AI for AIJ2023

Структура:
* sample_test - тест в нужно формате с сайта организаторов (из вкладки data)
* team_code - здесь весь код и модели команды
  * models
    * llm - mistral7b_instructed (https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/tree/main)
    * one-peace.pt - https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one-peace.pt
    * en_wiki_paragraphs.ann - индекс для RAG (https://disk.yandex.ru/d/Ankwc8Ic20mdYQ)
    * en_wiki_paragraphs.txt - сами параграфы для RAG (https://disk.yandex.ru/d/Ankwc8Ic20mdYQ)
    * wiki_onepeace_pca.pkl - понижение размерности для OP-эмбеддингов (https://disk.yandex.ru/d/Ankwc8Ic20mdYQ)
  * ONE-PEACE
    * ... - код либы
    * requirements.txt - нужно запустить отдельно, чтобы установить one-peace
  * generate.py - основные функции + служебные
* run.py - имитация скрипта для инференса модели
* requirements.txt - зависимости всего, кроме one-peace (ставить после установки one-peace)
* Dockerfile - наш докерфайл для загрузки образа в систему соревнования (подробнее - в инструкции во вкладке data)

!!! Для реального сабмита нужно поменять DEVICE в generate.py
