FROM nvidia/cuda:12.2.0-base-ubuntu22.04
LABEL maintainer "WincerChan <WincerChan@gmail.com>"

# Install OpenCL and Python dependencies in a single layer to reduce size
RUN apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 \
        python3-click \
        python3-base58 \
        python3-nacl \
        python3-numpy \
        python3-pyopencl && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    rm -rf /var/lib/apt/lists/*


# source codes
COPY core /app/core
COPY main.py /app
COPY LICENSE /app

# container-runtime
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV PYTHONUNBUFFERED=1

WORKDIR /app
ENTRYPOINT ["python3", "-u", "main.py"]
