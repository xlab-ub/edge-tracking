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

# Set working directory
WORKDIR /workspace

# Keep container running
CMD ["bash"]
