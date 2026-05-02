FROM docker.io/library/debian:trixie

ENV OPENCODE_INSTALL_DIR=/usr/local/bin
ENV PATH="/root/.opencode/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        coreutils \
        curl \
        findutils \
        git \
        less \
        nano \
        python3 \
        python3-pip \
        ripgrep \
        sed \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://opencode.ai/install | bash
RUN opencode --version

COPY entrypoint.sh /usr/local/bin/opencode-entrypoint.sh
RUN chmod +x /usr/local/bin/opencode-entrypoint.sh

WORKDIR /workspace

EXPOSE 4096

ENTRYPOINT ["/usr/local/bin/opencode-entrypoint.sh"]
