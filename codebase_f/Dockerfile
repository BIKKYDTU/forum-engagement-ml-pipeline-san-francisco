###############################################
# BASE IMAGE
###############################################
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

###############################################
# SYSTEM DEPENDENCIES
###############################################
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-setuptools \
    python-is-python3 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

###############################################
# PYTHON DEPENDENCIES (build-time install; no COPY/ADD of project files)
###############################################
RUN python3 -m pip install --no-cache-dir \
    pytest \
    "pandas>=2.0,<3.0" \
    "numpy>=1.24,<2.0" \
    "scikit-learn>=1.3,<2.0"

###############################################
# WORKING DIRECTORY + GIT SETUP
###############################################
WORKDIR /app

RUN git init \
    && git config --global user.email "agent@example.com" \
    && git config --global user.name "Agent" \
    && echo "# Workspace" > README.md \
    && git add README.md \
    && git commit -m "Initial commit"

###############################################
# EVALUATION ASSETS DIRECTORY
###############################################
RUN mkdir -p /eval_assets

CMD ["/bin/bash"]
