FROM mxnet/python:gpu

RUN apt-get update
RUN apt-get install -y python-dev
RUN pip2 install numpy==1.14.6 scipy sklearn pandas awscli boto3 spacy==2.0.11 -U
RUN python2 -m spacy download en_core_web_md

# add these files for batch prediction
ADD mxnet_model/utils.py /usr/local/bin/utils.py
ADD mxnet_model/predict.py /usr/local/bin/predict.py
ADD mxnet_model/train /usr/local/bin/train.py

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY mxnet_model /opt/program
WORKDIR /opt/program

# old
# WORKDIR /
USER root



 