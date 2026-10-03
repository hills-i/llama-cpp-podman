FROM node:26-trixie-slim

ARG PI_VERSION=1.0.0

ENV NPM_CONFIG_UPDATE_NOTIFIER=false \
    PI_OFFLINE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g --ignore-scripts @earendil-works/pi-coding-agent@${PI_VERSION} \
    && pi --version

RUN mkdir -p /root/.pi/agent
COPY models.json settings.json /root/.pi/agent/

WORKDIR /workspace

ENTRYPOINT ["pi"]
