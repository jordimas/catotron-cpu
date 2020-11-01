FROM python:3.7.9

COPY . /srv/

RUN pip install -U pip
RUN pip install -r /srv/requirements.txt


RUN mkdir /srv/models
WORKDIR /srv/models
RUN wget -q https://www.softcatala.org/pub/softcatala/catotron-models/upc_ona_tacotron2.pt
RUN wget -q https://www.softcatala.org/pub/softcatala/catotron-models/upc_pau_tacotron2.pt
RUN wget -q https://www.softcatala.org/pub/softcatala/catotron-models/melgan_onapau_catotron.pt

ENTRYPOINT ["python", "demo_server.py"]
