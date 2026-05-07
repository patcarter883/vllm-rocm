# syntax=docker/dockerfile:1-labs
ARG AMDGPU_FAMILY=gfx120X-all
ARG GPU_ARCH=gfx1201
ARG ROCM_VERSION=7.12.0a20260205

FROM ubuntu:24.04 AS base
ENV PYTHONUNBUFFERED=1
ARG AMDGPU_FAMILY
ARG GPU_ARCH
ARG ROCM_VERSION

SHELL ["/bin/bash", "-l", "-c"]
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libatomic1 \
    libgomp1 \
    libnuma-dev \
    wget && \
    rm -rf /var/lib/apt/lists/*

# setup venv and make the env active for all shell sessions,
# including run commands
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH=/root/.local/bin:$PATH
ENV PATH="/root/.local/bin:${PATH}"
RUN cd /app && uv venv --python 3.12 && \
    source .venv/bin/activate && \
    echo "source /app/.venv/bin/activate" > /root/.bash_profile



# install ROCm python packages
RUN uv pip install \
    --index-url https://rocm.nightlies.amd.com/v2/${AMDGPU_FAMILY}/ \
    "rocm[libraries, devel]" && \
    uv pip install \
    --index-url https://rocm.nightlies.amd.com/v2/${AMDGPU_FAMILY}/ \
    torch torchvision torchaudio

# install tarball of rocm
COPY rocm_tarballs/ /tmp/rocm_tarballs/
RUN mkdir -p /opt/rocm-$ROCM_VERSION && \
    tar xzf /tmp/rocm_tarballs/therock-dist-linux-$AMDGPU_FAMILY-$ROCM_VERSION.tar.gz -C /opt/rocm-$ROCM_VERSION && \
    rm -rf /tmp/rocm_tarballs/ && \
    ln -s /opt/rocm-$ROCM_VERSION /opt/rocm

ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=$ROCM_PATH/lib
# ENV CMAKE_PREFIX_PATH="/app/.venv/lib/python3.12/site-packages/torch/share/cmake/Torch"
ENV DEVICE_LIB_PATH=$ROCM_PATH/llvm/amdgcn/bitcode  
ENV HIP_DEVICE_LIB_PATH=$ROCM_PATH/llvm/amdgcn/bitcode
# Note: Do NOT set FLASH_ATTENTION_TRITON_AMD_ENABLE for CK backend (we need CK, not Triton)
# FLASH_ATTENTION_TRITON_AMD_ENABLE is not set so CK backend is used for gfx12
ENV PYTORCH_ROCM_ARCH=${GPU_ARCH}
ENV PATH=${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}
ENV CC=$ROCM_PATH/llvm/bin/clang
ENV CXX=$ROCM_PATH/llvm/bin/clang++
ENV HIPCC=$ROCM_PATH/bin/hipcc
ENV VLLM_TARGET_DEVICE="rocm"
ENV GPU_TARGETS="${GPU_ARCH}"
ENV Torch_DIR="/app/.venv/lib/python3.12/site-packages/torch/share/cmake/Torch"

# copy .bash_profile to .bashrc
RUN cp /root/.bash_profile /root/.bashrc

# Fix hardcoded paths in ROCm torch cmake files (from TheRock builds)
# The torch package from ROCm nightlies has cmake config files referencing
# build machine paths like:
#   .../rocm_sysdeps/lib/pkgconfig/../../include
# which don't exist in the container.
# Create intermediate directories so path resolution works with ../.. traversal,
# then symlink the final include dir to /opt/rocm/include
RUN mkdir -p /therock/output/build/third-party/sysdeps/linux/libdrm/build/stage/lib/rocm_sysdeps/lib/pkgconfig && \
    ln -sfn /opt/rocm/include /therock/output/build/third-party/sysdeps/linux/libdrm/build/stage/lib/rocm_sysdeps/include

# clone vllm
RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && git checkout v0.20.1 && \
    # Install numpy<2 FIRST before any other packages to ensure version constraint
    uv pip install "numpy<2" && \
    # Upgrade build tools before installing rocm requirements
    uv pip install --upgrade numba \
        scipy \
        cmake \
        "setuptools-scm>=8" && \
    python use_existing_torch.py && \
    uv pip install -r requirements/rocm.txt && \
    python setup.py develop && \
    # amd_smi is included in TheRock tarball at /opt/rocm/share/amd_smi
    uv pip install /opt/rocm/share/amd_smi || echo "amd_smi not found, continuing..."

# flash-attention with gfx12 support via CK (composable kernel) backend
# ninja is required for faster parallel compilation (installed via setup_requires)
# GPU_ARCHS is required for CK to generate gfx12 kernels during build
# Note: Do NOT set FLASH_ATTENTION_TRITON_AMD_ENABLE as we want CK backend for gfx12
RUN git clone https://github.com/hyoon1/flash-attention.git && \
    cd flash-attention && \
    git checkout enable-ck-gfx12 && \
    git submodule update --init --recursive csrc/composable_kernel csrc/cutlass && \
    GPU_ARCHS="gfx1201" python setup.py install

ENTRYPOINT [ "/app/.venv/bin/vllm","serve"]