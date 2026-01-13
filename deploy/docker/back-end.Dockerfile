# docker build --build-arg BASE_IMAGE="" -build-arg ENVIRONMENT=""
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV base_image=${BASE_IMAGE}

# ENV PYDEVD_DISABLE_FILE_VALIDATION=1

# optional environment: development、test、production;  From 为变量作用域
ARG ENVIRONMENT
ENV environment=${ENVIRONMENT}

ARG APP_PATH
ENV app_path=${APP_PATH}

ADD ./ /code
WORKDIR /code

RUN mkdir static
# RUN python -m compileall -b .
# RUN find . -name "*.py" -type f -delete

EXPOSE 8000
# ENTRYPOINT ["sh", "-c", "python apis/entrypoint/main.pyc ${app_path}"]
ENTRYPOINT ["sh", "-c", "python apis/entrypoint/main.py ${app_path}"]
