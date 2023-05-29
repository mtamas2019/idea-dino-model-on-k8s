# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

# Disable interactive
ENV DEBIAN_FRONTEND noninteractive

# Update and upgrade Ubuntu packages
RUN apt-get update && apt-get upgrade -y

#Install necessary packages
RUN apt-get install -y libsm6 libxext6 libxrender-dev apt-utils curl git

# Install Miniconda
ENV MINICONDA_VERSION 4.10.3
ENV CONDA_DIR /opt/conda
RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_${MINICONDA_VERSION}-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p $CONDA_DIR \
    && rm ~/miniconda.sh \
    && echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh \
    && echo "source $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc
ENV PATH $CONDA_DIR/bin:$PATH

# Install python packages
RUN conda install python=3.7.3

RUN conda install numpy

RUN conda install pytorch=1.9.0 torchvision=0.10.0 pytorch-cuda=11.7 -c pytorch -c nvidia

RUN pip install termcolor addict cython submitit yapf pycocotools timm scipy

RUN pip install Pillow fastapi uvicorn  python-multipart

# Set the working directory
WORKDIR /app

# Copy inference files
COPY inference.tgz /app

RUN tar -xzf inference.tgz -C /app

# Build MultiAttention
WORKDIR /app/inference/ops

RUN bash make.sh

# Copy Checkpoint
WORKDIR /app/inference

COPY checkpoint0023_4scale.pth /app/inference/assets/

# Set the environment variables
ENV MODEL_CONFIG_PATH assets/DINO_4scale.py

ENV MODEL_CHECKPOINT_NAME assets/checkpoint0023_4scale.pth

# Set the entry point for the container
ENTRYPOINT ["python", "inference.py"]
