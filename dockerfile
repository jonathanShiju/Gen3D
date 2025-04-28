# Use the official PyTorch image with CUDA 12.4 and cuDNN 9 for GPU acceleration
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

# Set the working directory to the root
WORKDIR /

# Set the CUDA architecture for PyTorch, targeting various GPU compute capabilities
ENV TORCH_CUDA_ARCH_LIST=8.0;8.6;8.7;8.9

# Install essential system packages and libraries needed for building and running Python packages
RUN apt update \
 && apt-get install -yq --no-install-recommends \
    git \
    curl \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    libxext6 \
    build-essential \
    libgl1-mesa-dev \
    libx11-dev \
    libxi-dev \
    libxext-dev \
    libxxf86vm-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libopenexr-dev \
    libomp-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install basic Python libraries required for image processing, 3D visualization, and other functionalities
RUN pip install \
    pillow \
    imageio \
    imageio-ffmpeg \
    tqdm \
    easydict \
    opencv-python-headless \
    scipy \
    ninja \
    onnxruntime \
    trimesh \
    xatlas \
    pyvista \
    pymeshfix \
    igraph \
    transformers \
    rembg \
    pydantic 

RUN pip install torch==2.4.1+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124

# Install utils3d from a specific Git commit
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Install Open3D library, ignoring previously installed versions
RUN pip install --ignore-installed open3d

# Install xformers, Flash Attention, and nvdiffrast libraries from external sources
RUN pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install flash-attn && \
    pip install git+https://github.com/NVlabs/nvdiffrast.git

# Install Kaolin, a library for deep learning on 3D data, for PyTorch 2.4.1 with CUDA 12.4
RUN pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu124.html

# Set environment variables for CUDA and C++17 flags for building certain packages
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV CXXFLAGS="-std=c++17"

# Clone and install DiffOctreeRasterization extension
RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast \
 && pip install /tmp/extensions/diffoctreerast

# Clone and install MIP Splatting extension
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting \
 && pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# Copy and install Vox2Seq extension
COPY extensions/vox2seq /tmp/extensions/vox2seq
RUN pip install /tmp/extensions/vox2seq

# Install SpConv (sparse convolution library) for CUDA 12.0 and Jupyter notebook
RUN pip install spconv-cu120 

# Set the working directory to /workspace
WORKDIR /workspace

# Copy Trellis-related files into the container
COPY trellis ./trellis 
COPY TRELLIS-image-large ./TRELLIS-image-large
COPY Trellis_i23D_ModelGeneration.py .
COPY Trellis_ModelGeneration.py .
COPY TRELLIS-text-base ./TRELLIS-text-base
COPY TRELLIS-text-large ./TRELLIS-text-large
COPY TRELLIS-text-xlarge ./TRELLIS-text-xlarge
COPY Trellis_t23D_ModelGeneration.py .

# RUN useradd -ms /bin/bash user 
# # Change ownership of the app files to the non-root user
# RUN chown -R user:user /workspace
# USER user
# RUN mkdir -p /home/user/.cache

# # Create a directory for volumes and declare the volume for external access
# RUN mkdir /workspace/Volume
# VOLUME ["/workspace/Volume"]
