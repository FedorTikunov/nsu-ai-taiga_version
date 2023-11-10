FROM cr.msk.sbercloud.ru/aicloud-base-images-test/cuda11.7-torch2:fdf9bece-630252

USER root

WORKDIR /app

COPY . ./

WORKDIR /app/team_code/ONE-PEACE
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

USER jovyan
WORKDIR /home/jovyan