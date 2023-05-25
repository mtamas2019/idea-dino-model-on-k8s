# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

# Define environment var for detect GPUs
ENV DEBIAN_FRONTEND noninteractive

ENV DATASET_DIR /opt/ml/input/data/coco2017

# Update and upgrade Ubuntu packages
RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y git curl apt-utils

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

RUN conda install python=3.7.3

RUN conda install numpy

RUN conda install pytorch=1.9.0 torchvision=0.10.0 pytorch-cuda=11.7 -c pytorch -c nvidia

RUN pip install termcolor addict cython submitit yapf pycocotools timm scipy

WORKDIR /opt/ml/code

RUN git clone https://github.com/IDEA-Research/DINO.git

COPY make.sh /opt/ml/code/DINO/models/dino/ops/

WORKDIR /opt/ml/code/DINO/models/dino/ops

RUN bash make.sh

COPY DINO_train.sh /opt/ml/code/DINO/scripts

WORKDIR /opt/ml/code/DINO/

RUN chmod +X scripts/DINO_train.sh

#RUN apt-get install -y nvidia-utils-515-server

ENTRYPOINT bash scripts/DINO_train.sh $DATASET_DIR
