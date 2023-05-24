FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y software-properties-common nvidia-utils-515-server git

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.7 python3-pip

RUN pip3 install --upgrade pip

RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

COPY requirements.txt /install/

COPY MultiScaleDeformableAttention-1.0-py3-none-any.whl /install/

WORKDIR /install

RUN pip3 install -r requirements.txt

RUN pip3 install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI&egg=pycocotools

WORKDIR /opt/ml/code

RUN git clone https://github.com/IDEA-Research/DINO.git

WORKDIR /opt/ml/code/DINO

RUN chmod +x scripts/DINO_train.sh

RUN ln -s /usr/bin/python3 /usr/bin/python

CMD ["bash", "-c", "./scripts/DINO_train.sh \"$DATASET_DIR\""]
