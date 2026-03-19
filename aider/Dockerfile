FROM docker.io/library/python:3.12

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -U "aider-chat[browser]"

COPY entrypoint.sh /usr/local/bin/aider-entrypoint.sh
RUN chmod +x /usr/local/bin/aider-entrypoint.sh

WORKDIR /workspace

EXPOSE 8501

ENTRYPOINT ["/usr/local/bin/aider-entrypoint.sh"]
