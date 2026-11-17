# One image for all co-authors: the full bp26 env (CPU or GPU) and Claude Code, isolated from the host OS.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# build-essential: fallback if a pinned package (e.g. tick) has no wheel and must compile
RUN apt-get update && apt-get install -y --no-install-recommends \
        git tmux curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pinned project packages, the same list environment.yml installs locally
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Claude Code, installed globally so both `dev` and root (GPU server) can run it
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# Non-root user for co-authors; VS Code remaps its UID to the host user (updateRemoteUserUID)
RUN useradd -m -s /bin/bash dev \
    && mkdir -p /home/dev/.claude && chown dev:dev /home/dev/.claude
