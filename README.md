# NSU-AI for AIJ2023

### Концептуальное описание алгоритма:

1. Вопрос, заданный только в текстовой модальности, сразу отправляется на большую языковую модель как часть подводки (промпта).
2. Данные в звуковой или зрительной модальностях (то есть аудио-сигнал или картинка) преобразуются в текст и встраиваются в текстовую подводку следующим образом:
    - трёхмодальный эмбеддер ONE PEACE генерирует семантический вектор по звуку или картинке (так называемый OP-эмбеддинг);
    - размерность OP-эмбеддинга понижается с помощью анализа главных компонент (PCA); 
    - по векторной базе (по специальному Annoy-индексу) производится поиск 100 текстов (параграфов из Википедии), чьи вектора наиболее похожи на входной OP-эмбеддинг;
    - для этой же картинки или звука генерируется краткое текстовое описание с помощью вспомогательных моделей:
      - если это картинка, то специальный генератор подписей для картинок генерирует её подпись;
      - если это неречевой звук, то классификатор звуков определяет класс этого звука согласно онтологии [Audioset](https://research.google.com/audioset);
      - если это речь, то происходит её распознавание с помощью дистиллированного Whisper-Medium;
    - топ-100 найденных параграфов из Википедии переранжируются по возрастанию косинусного расстояния между их sentence-эмбеддингами и sentence-эмбеддингом краткого описания (в качестве генератора эмбеддингов используется MPNet-Base из библиотеки Sentence Transformers);
    - итоговое текстовое описание любого объекта в не-текстовой модальности, кроме речи, формируется путём конкатенации краткого текстового описания и наиболее релевантного (после переранжирования) параграфа из Википедии; для речи же просто используется результат распознавания через Whisper. 

### Структура проекта:

* `sample_test` - тест в нужном формате с сайта организаторов (из вкладки data)
* `team_code` - здесь весь код и модели команды
  * `models`
    * `llm` - большая языковая модель Mistral-7b-instruct (https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/tree/main);
    * `one-peace.pt` - трёхмодальный (текст, картинки, звук) эмбеддер https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one-peace.pt;
    * данные для мультимодального RAG: annoy-индекс `en_wiki_paragraphs.ann`, сами параграфы `en_wiki_paragraphs.txt` и sklearn-пайплайн понижения размерности `wiki_onepeace_pca.pkl` в виде **пятитомного архива**:
      1. **часть 1** https://disk.yandex.ru/d/R2cglcaxVq2cRQ;
      2. **часть 2** https://disk.yandex.ru/d/ohta4v4EcX8DEw;
      3. **часть 3** https://disk.yandex.ru/d/lnz30eszyqEPaA;
      4. **часть 4** https://disk.yandex.ru/d/6uPE8U31aceSWA;
      5. **часть 5** https://disk.yandex.ru/d/7qIYujlqdsRXZA;
    * `auxiliary_models` - подкаталог с четырьмя вспомогательными моделями:
      1. `audioset` - классификатор аудио (https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593);
      2. `blip` - генератор подписей для картинок (https://huggingface.co/Salesforce/blip-image-captioning-base);
      3. `sbert` - генератор эмбеддингов предложений для переранжирования результатов RAG по косинусному расстоянию (https://huggingface.co/sentence-transformers/all-mpnet-base-v2);
      4. `whisper_medium` - распознаватель английской речи (https://huggingface.co/distil-whisper/distil-medium.en)
  * `ONE-PEACE`
    * ... - код либы
    * `requirements.txt` - нужно запустить отдельно, чтобы установить one-peace
  * `generate.py` - основные функции + служебные
* `run.py` - имитация скрипта для инференса модели
* `requirements.txt` - зависимости всего, кроме one-peace (ставить после установки one-peace)
* `Dockerfile` - наш докерфайл для загрузки образа в систему соревнования (подробнее - в инструкции во вкладке data)

### Примечания

!!! Для реального сабмита нужно поменять DEVICE в generate.py


docker login cr.msk.sbercloud.ru  # aijcontest AIJcontest0000

docker pull cr.msk.sbercloud.ru/aicloud-base-images-test/cuda11.7-torch2:fdf9bece-630252

docker build --tag nsuaiaij:0.1 .
docker tag nsuaiaij:0.1 cr.msk.sbercloud.ru/aijcontest/nsu-ai:0.1
docker push cr.msk.sbercloud.ru/aijcontest/nsu-ai:0.1