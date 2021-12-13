FROM nvcr.io/nvidia/tensorflow:21.08-tf2-py3

RUN apt update 

WORKDIR /workspace

COPY pyproject.toml .

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl git build-essential \
    && pip install poetry \ 
    && poetry config virtualenvs.create false \
    && poetry install \
    && rm pyproject.toml
