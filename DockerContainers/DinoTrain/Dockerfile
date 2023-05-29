# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

# Disable apt interactive mode
ENV DEBIAN_FRONTEND noninteractive

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

# Install python packages
RUN conda install python=3.7.3

RUN conda install numpy

RUN conda install pytorch=1.9.0 torchvision=0.10.0 pytorch-cuda=11.7 -c pytorch -c nvidia

RUN pip install termcolor addict cython submitit yapf pycocotools timm scipy

WORKDIR /opt/ml/code

# Git clone DINO source
#RUN git clone https://github.com/IDEA-Research/DINO.git

RUN git clone https://github.com/mtamas2019/DINO.git

# Build cuda code in build time
COPY make.sh /opt/ml/code/DINO/models/dino/ops/

WORKDIR /opt/ml/code/DINO/models/dino/ops

RUN bash make.sh

WORKDIR /opt/ml/code/DINO/

# Define environment variables
ENV DATASET_DIR /opt/ml/input/data/coco2017
ENV TRAIN_OUTPUT_DIR logs/DINO/R50-MS4 
ENV CONFIG_FILE config/DINO/DINO_4scale.py
ENV WORKERS 1
ENV NUM_CLASSES 91
ENV LEARNING_RATE 0.0001
ENV BATCH_SIZE 2
ENV EPOCHS 12
ENV USE_CHECKPOINT TRUE
ENV USE_EMA False
ENV DN_SCALAR 100
ENV EMBED_INIT_TGT True
ENV DN_LABEL_COEF 1.0
ENV DN_BBOX_COEF 1.0
ENV DN_BOX_NOISE_SCALE 1.0

# Run traning code
ENTRYPOINT python main.py --output_dir $TRAIN_OUTPUT_DIR -c $CONFIG_FILE --coco_path $DATASET_DIR --num_workers $WORKERS --save_log \
                    --options dn_scalar=$DN_SCALAR embed_init_tgt=$EMBED_INIT_TGT dn_label_coef=$DN_LABEL_COEF dn_bbox_coef=$DN_BBOX_COEF use_ema=$USE_EMA \
                      dn_box_noise_scale=$DN_BOX_NOISE_SCALE  num_classes=$NUM_CLASSES lr=$LEARNING_RATE batch_size=$BATCH_SIZE \
                      epochs=$EPOCHS use_checkpoint=$USE_CHECKPOINT
