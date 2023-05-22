# syntax=docker/dockerfile:1
   
FROM nvidia/cuda:11.7.0-devel-ubuntu20.04
COPY test-cuda.cu /app/test-cuda.cu
WORKDIR /app
RUN nvcc test-cuda.cu -o test-cuda
ENTRYPOINT ["./test-cuda"]
