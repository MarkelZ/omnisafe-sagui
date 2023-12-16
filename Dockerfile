# Set python 3.10
FROM python:3.10.12

# ENV DEBIAN_FRONTEND=noninteractive
# SHELL ["/bin/bash", "-c"]

# Create a virtual environment
RUN python3.10 -m venv venv

# Copy repo to the container
WORKDIR /omnisafe-sagui/
COPY . /omnisafe-sagui/

# Install packages
RUN apt-get update && \
    apt-get install -y sudo ca-certificates openssl \
    vim \
    libopenmpi-dev \
    git ssh build-essential gcc g++ cmake make \
    python3-dev python3-venv python3-opengl libosmesa6-dev && \
    rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8
ENV MUJOCO_GL osmesa
ENV PYOPENGL_PLATFORM osmesa
ENV CC=gcc CXX=g++

RUN python -m pip install --upgrade pip && \
    python -m pip install wheel && \
    python -m pip install -r requirements.txt && \
    rm -rf ~/.pip/cache ~/.cache/pip

RUN python -m pip install -e .

# Use a shell as the entry point
ENTRYPOINT ["sh", "-c", "git fetch && git pull && /bin/bash"]
