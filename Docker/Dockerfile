FROM ultralytics/ultralytics:latest-jetson-jetpack6

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}

# Install any additional packages you need
RUN apt-get update && apt-get install -y \
    vim \
    git \
    nano \
    gedit \
    x11-apps \
    x11-utils \
    v4l-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for missing libraries if needed
RUN ldconfig

# Set working directory
WORKDIR /workspace

# Keep container running
CMD ["bash"]