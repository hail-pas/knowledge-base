FROM python:3.11-slim-bullseye

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# RUN apt-get update
# ssh
# RUN apt-get install -y openssh-client
# db geo
# RUN apt-get install -y build-essential make cmake
# RUN apt-get install -y zlib1g-dev libgeos-dev libgeos-c1v5

RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone

ADD ./pyproject.toml /base/pyproject.toml
ADD ./uv.lock /base/uv.lock
WORKDIR /base

RUN uv sync --frozen
