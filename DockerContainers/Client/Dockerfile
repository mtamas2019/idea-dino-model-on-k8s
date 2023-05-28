FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# Update the package lists
RUN apt-get update

# Install Python 3.8 and pip
RUN apt-get install -y python3.8 python3-pip

# Set the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Install required packages
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

Label rebuid=1
# Copy the remaining files into the container
COPY client.tgz /app

# Extract the client.tgz file
RUN tar -xzf client.tgz -C /app

WORKDIR /app/client

# Set the environment variables
ENV API_URL="http://localhost:8080/inference"
ENV PORT=80

# Set the entry point for the container
ENTRYPOINT ["python3", "client.py"]
