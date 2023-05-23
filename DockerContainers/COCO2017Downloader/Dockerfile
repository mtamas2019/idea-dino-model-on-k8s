# Use Python 3.7.3 as the base image
FROM python:3.7.3

# Set the working directory in the container
WORKDIR /opt/ml/code

# Copy the Python script to the working directory
COPY download_dataset.py /opt/ml/code

# Copy the requirements.txt  to the working directory
COPY requirements.txt /opt/ml/code

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point
ENTRYPOINT ["python", "download_dataset.py"]
