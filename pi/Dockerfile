FROM node:26-trixie-slim

ENV NPM_CONFIG_UPDATE_NOTIFIER=false \
    PI_OFFLINE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g --ignore-scripts @earendil-works/pi-coding-agent \
    && pi --version

RUN mkdir -p /root/.pi/agent
COPY models.json /root/.pi/agent/models.json

WORKDIR /workspace

ENTRYPOINT ["pi"]
