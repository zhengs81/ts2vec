FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

# Set Aliyun mirror source for Ubuntu package repositories
RUN echo \
"deb https://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse\n\
deb-src https://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse\n\
deb https://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse\n\
deb-src https://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse\n\
deb https://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse\n\
deb-src https://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse\n\
deb https://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse\n\
deb-src https://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse\n\
deb https://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse\n\
deb-src https://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse" > /etc/apt/sources.list

# Update package manager and install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    libffi-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.9.7 from source
RUN wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz && \
    tar -xf Python-3.9.7.tgz && \
    cd Python-3.9.7 && \
    ./configure --enable-optimizations && \
    make -j "$(nproc)" && \
    make altinstall

# Clean up Python source files
RUN rm -rf Python-3.9.7 Python-3.9.7.tgz

# Set Python 3.9 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.9 1

# Set up pip and install packages from requirements.txt
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN mkdir /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

WORKDIR /app
# Your additional Dockerfile commands go here

# Specify the command to run the application, if applicable
ENTRYPOINT ["python3", "train.py"]