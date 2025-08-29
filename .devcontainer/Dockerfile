FROM ultralytics/ultralytics:latest-jetson-jetpack6

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

# Fix CUDA library paths for CUDA 12.6
RUN mkdir -p /usr/local/cuda/lib64 && \
    ln -sf /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublas.so.12 /usr/local/cuda/lib64/libcublas.so.12 && \
    ln -sf /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublas.so.12 /usr/local/cuda/lib64/libcublas.so.11 && \
    ln -sf /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudnn.so.9 /usr/local/cuda/lib64/libcudnn.so.9 2>/dev/null || true

# Create a dummy libnvToolsExt.so.1 (it's deprecated in CUDA 12.6)
RUN touch /usr/local/cuda/lib64/libnvToolsExt.so.1 && \
    chmod 755 /usr/local/cuda/lib64/libnvToolsExt.so.1

# Set up CUDA environment variables
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda

# Update library cache
RUN ldconfig

# Set working directory
WORKDIR /workspace

# Keep container running
CMD ["bash"]