# **Project information**

This project contains a description of the steps involved in the containerization process of the IDEA Research DINO object-detecting vision transformer model. 

IDEA Research DINO project page: https://github.com/IDEA-Research/DINO

**Created containers**

*   CUDA test container image, for testing CUDA code compilation in a container. https://hub.docker.com/r/mtamasdocker/cuda-test-11-7/tags

*   Coco2017 dataset downloader container image. Dataset website: https://cocodataset.org/#home,
    Image: https://hub.docker.com/r/mtamasdocker/coco2017-dataset-downloader/tags

*  Training container. Image: https://hub.docker.com/r/mtamasdocker/idea-research-dino-train/tags

*  Inference Service, based on Flask API. 
   Image: https://hub.docker.com/r/mtamasdocker/idea-research-dino-inference/tags

* Client app with web UI for inference testing.
  Image: https://hub.docker.com/r/mtamasdocker/idea-research-dino-client-app/tags

**Infrasturcutre:**

I wanted to use AWS public cloud, but they did not accept my request for GPU-enabled EC2, so the project was finally implemented in Genesis Cloud. Genesis Cloud is cheaper, however, it significantly limits the possibilities, flexibility, and has lots of limitations.
Website [genesiscloud.com](https://genesiscloud.com)

I used an RTX 3060 TI GPU (8 GB memory), except during training when I used an RTX 3090 (24 GB memory), because the memory of the 3060 TI was insufficient.

## 1. Getting familiar with DINO

As a first step, I familiarized myself with the DINO model and a little bit with Pytorch framework

After reading through the documentation, I tried out the model on Ubuntu without containerization. I tested the evaluation and training mentioned in the documentation.

I used conda environment and package manager.

**Used versions:**
 * OS: Ubuntu 20.04 (Image from genesis cloud)
 * Nvidia driver: nvidia-driver-515-server (from Ubuntu package manager)
 * Cuda Toolkit: 11.7 (https://developer.nvidia.com/cuda-11-7-0-download-archive)
 * Conda: 23.3.1
 * Python: 3.7.3
 * pytorch: 1.9.0
 * pytorch-cuda: 11.7 (The version 11.1 mentioned in the documentation is no longer available.)
 * torchvision: 0.10.0
 * numpy: 1.21.5


 **nvidia-smi output**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.105.01   Driver Version: 515.105.01   CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:00:05.0 Off |                  N/A |
|  0%   32C    P0    34W / 200W |      0MiB /  8192MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                
```
      
**Evaulation result:**

```
Accumulating evaluation results...
DONE (t=38.39s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.504
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.683
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.548
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.334
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.538
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.648
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.383
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.659
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.731
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.572
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.773
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.882

 ```

## 2. Deformable-Convolution-V2-PyTorch

This had to be compiled from CUDA code.  I wanted to create a python package from it, so that during the creation of containers, I don't have to deal with the compilation.

The package creation was successful, it was also installed in the container, but when running, I got a missing attribute error when the code wanted to use this model, package. Due to this, the solution remained that this model was compiled at container build time.

The Python package versions matched during compilation and running, probably the problems arose from the discrepancy between the OS image provided by Genesis Cloud and the image used by the container.

Here are the files needed to create the Python package (but this not working, mentioned above): https://github.com/mtamas2019/idea-dino-model-on-k8s/tree/main/Misc/MultiScaleDeformableAttention

## 3. Kubernetes environment setup

I created a GPU-supported Kubernetes environment with Minikube using the following documents:

* Nvidia container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide

* Nvidia container runtime: https://github.com/NVIDIA/nvidia-container-runtime#installation

* Minikube install: https://minikube.sigs.k8s.io/docs/start/

* Minikube GPU support: https://minikube.sigs.k8s.io/docs/tutorials/nvidia_gpu/

You can use this Ansible playbooks to install nvidia gpu supported docker environment: https://github.com/mtamas2019/idea-dino-model-on-k8s/tree/main/Ansible

I used Minikube with none driver and kubernetes 1.23. I used 1.23 old k8s version, because more things have changed in the newer versions, more packages and configurations are needed for it and I wanted to save time with this, because the main focus of the project was not to run the environment on the latest version of k8s

```
 minikube start --driver=none --kubernetes-version=v1.23.0
```

**It's important to note** that starting minikube overwrites the /etc/docker/daemon.json file that was properly set according to the previous nvidia documentation. This needs to be checked and corrected.


I tested the GPU support of the environment with a container that compiles CUDA code in runtime (Note: this container image not test the pytorch cuda compatibility). 

Cuda test image: https://hub.docker.com/r/mtamasdocker/cuda-test-11-7


## 4. Create dataset downloader container

As a first step, I created a python script that downloads the coco2017 dataset. Then I containerized it. In Kubernetes, I created a job for it as well as a persistent volume.

I used Python 3.7.3 version to ensure the python version is consistent in all containers. This is the version that IDEA-Research tested the DINO model on.

* Docker image source: https://github.com/mtamas2019/idea-dino-model-on-k8s/tree/main/DockerContainers/COCO2017Downloader

* Docker image: https://hub.docker.com/r/mtamasdocker/coco2017-dataset-downloader/tags

* Kubernetes manifests:

    * https://github.com/mtamas2019/idea-dino-model-on-k8s/tree/main/k8s-manifests/train

    * https://github.com/mtamas2019/idea-dino-model-on-k8s/tree/main/k8s-manifests/volumes

## 5. Create parametrized training container

Next, I created the training container. I wanted to do this essentially without 
conda, but this solution did not work due to outdated packages that are no longer available or package compatibility issues. Because of this, I remained with the proven conda solution from the non-container version, with the only change being that I used the Nvidia base image here with cuda 11.7.

I have selected quite a few parameters that will be parameterized from k8s via environment variables.

I probably spent the most time with this part, to get the right package versions and compatibilities lined up. Also, to get the CUDA code to compile at build time. The container image wasn't small, there's probably room for optimization.

Example build command:
```
DOCKER_BUILDKIT=0 docker build -t dinotrain:1.0 . --network=host
```
**Important:** If the cuda code compilation fails check the /etc/docker/daemon.json file that was properly set according to the nvidia documentation. 

Following this, I created the appropriate k8s manifest files. Here, I also used a job. Furthermore, two persistent volumes. One contains the dataset downloaded by the previous job, the other contains the directory of the training output. Checkpoints, logs, and config file dumps go into the output directory.

Due to the small amount of shared memory in minikube, I had to give the container a temporary file-based memory area here.

In this example project, I did not use distributed training, I only had one GPU available.

*  Docker container source: https://github.com/mtamas2019/idea-dino-model-on-k8s/tree/main/DockerContainers/DinoTrain

*  Docker images: 
   https://hub.docker.com/r/mtamasdocker/idea-research-dino-train/tags

   (Note: 1.0: contains origional IDE-Research DINO source, 1.1 image version contains my DINO fork source code with pythorch profiler)

* Kubernetes manifests:
   * https://github.com/mtamas2019/idea-dino-model-on-k8s/tree/main/k8s-manifests/train
   
   * https://github.com/mtamas2019/idea-dino-model-on-k8s/tree/main/k8s-manifests/volumes

Note:
I planned to put the dataset downloader and the train containers into one job, and the downloader would be the initcontainer, but eventually they stayed separate, it's probably better to solve this at the workflow management level.


## 6. Model train

In the next step, I started the training. In the first round, it turned out that the CUDA code compiled in the environment without a container is not operational, so its compilation was included in the Dockerfile.

After solving this, the next problem occurred, which was the lack of GPU memory. I solved this by swapping the GPU in a virtual machine, and from then on, I started the training on an RTX 3090. Then the training started properly.

When I started the training, I saw that it takes 1.5 days for 1 epoch to complete on this GPU.

![Training](https://github.com/mtamas2019/idea-dino-model-on-k8s/blob/b1e371b3c49753eda7a60cfa627c28c2c985f024/Misc/documentation/training.jpg?raw=true "Traning")

I wanted to use Tensorboard for analysis, but it would have required running a full epoch. This seemed to be costly considering the 1.5-day runtime, so it didn't happen, but despite this, the PyTorch profiler part was included in the 1.1 training image after a fork of original code.

My github repository with profiler: https://github.com/mtamas2019/DINO

Finally, I ran the training for about 1.5 hours, and the logs of it can be found here: https://github.com/mtamas2019/idea-dino-model-on-k8s/tree/main/RunResults/1

There was limited data available, so I created a simple python analysis script that reads a few parameters from the log file into a pandas dataframe and plots two values: the loss and the class_error.

Analysis script: https://github.com/mtamas2019/idea-dino-model-on-k8s/blob/main/Misc/analysis/analysis.py

The trend of the loss value already shows a decrease, also minimally in the case of class_error, but this training time was not enough for significant change

![Loss](https://raw.githubusercontent.com/mtamas2019/idea-dino-model-on-k8s/main/Misc/analysis/loss_plot.png)

![class_error](https://raw.githubusercontent.com/mtamas2019/idea-dino-model-on-k8s/main/Misc/analysis/class_error_plot.png "class_error")

## 7. Inference service

I created an inference service based on FastAPI, which accepts an image at the /inference API endpoint and after prediction, it returns the coordinates of the boxes, the ids of the classes, and the probabilities.
I used uvicorn, so the code can serve multiple requests at the same time.

I wanted to create a lightweight container, but this again required the CUDA code, so I remained with the tried conda solution in the end.

The GitHub repo does not include the checkpoint file due to its large size, it currently gets added to the container at build time, but it can be parameterized from an environment variable, so another checkpoint can be used, for example from a k8s persistent volume.

The k8s manifest files were also prepared. For the inference service, a deployment and a service yaml file were created.

*  Docker container source: https://github.com/mtamas2019/idea-dino-model-on-k8s/tree/main/DockerContainers/InferenceService

*  Docker images: 
   https://hub.docker.com/r/mtamasdocker/idea-research-dino-inference/tags

* Kubernetes manifests:
   * https://github.com/mtamas2019/idea-dino-model-on-k8s/tree/main/k8s-manifests/inference

## 8. Client application

 As a final step, a Flask client application with a web interface was created

 It can be parameterized through environment variables to determine which port it listens on, what the inference service API endpoint is, and the threshold for probabilities.

 An image can be uploaded on the client application's web interface, which is then sent to the inference service. The received values are then drawn on the original image and displayed.

 The k8s manifest files were also prepared for the client app.

 The Ingress controller was not installed in the Kubernetes environment, so in this case, the web interface can be accessed on port 30000.

**A short demonstration video showcasing the functionality:**
https://drive.google.com/file/d/1b5ayFgTmlTVklUbne5YY5a9gw4LBi79L/view?usp=share_link

* Docker container source: https://github.com/mtamas2019/idea-dino-model-on-k8s/tree/main/DockerContainers/Client

* Docker images: https://hub.docker.com/r/mtamasdocker/idea-research-dino-client-app/tags

(Note: in the 1.1 version, you can modify the probability threshold with a TRESHOLD env variable)

* Kubernetes manifests:
https://github.com/mtamas2019/idea-dino-model-on-k8s/tree/main/DockerContainers/Client

## 9. Possible further improvments, developments

*  Creating Ingress component for the k8s environment
*  Creating a Helm chart from the k8s manifests
*  Optimize container image sizes
*  Creating an Ansible playbook for establishing a GPU-supported Ubuntu 20.04 and k8s or use AWS (userdata or Deep Learning images)
*  Developing an ML workflow using Kubeflow or MLflow




