# syntax=docker/dockerfile:1.7

#  Copyright (c) 2026. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

# sudo docker buildx build --platform linux/amd64 --tag akihikokayaba/py3.12-torch2.11-cu12.6 --push . 2>&1 | tee /win/flounder/user/koku/temp/build.log


FROM ghcr.io/astral-sh/uv:latest AS uv


FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS build-base
LABEL authors="gu"

ARG DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:/usr/lib/ccache:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev \
        ninja-build \
        openssl \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN /usr/sbin/update-ccache-symlinks && \
    mkdir -p /opt/ccache && \
    ccache --set-config=cache_dir=/opt/ccache

COPY --from=uv /uv /uvx /usr/local/bin/
RUN uv python install 3.12 && \
    uv venv --python 3.12 "$VIRTUAL_ENV"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        pip \
        wheel \
        numpy \
        cupy-cuda12x \
        cucim-cu12 \
        cutensor-cu12 \
        nvidia-nccl-cu12 \
        mypy \
        ninja \
        psutil \
        setuptools \
        toml \
        pyarrow \
        jupyter \
        jupyterlab \
        importlib_metadata \
        regex \
        python-dotenv \
        omegaconf \
        yacs \
        ftfy \
        pytest \
        tensorboard \
        tensorboardX \
        pyinstaller


FROM build-base AS build-torch
ARG TARGETARCH
ARG FORCE_CUDA=1
ARG TORCH_CUDA_ARCH_LIST="5.2 6.1 7.0 7.2 7.5 8.0 8.6 8.9 9.0"
ENV CUDA_HOME=/usr/local/cuda

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        torch==2.11.0 \
        torchvision==0.26.0 \
        torchaudio==2.11.0 \
        torch-tensorrt==2.11.0+cu126 \
        tensorrt-cu12 \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        --index-strategy unsafe-best-match

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        triton \
        timm \
        nvidia-modelopt[all] \
        onnx \
        onnxscript \
        onnxruntime-gpu \
        vision-aided-loss \
        "lightning[extra]" \
        accelerate \
        transformers \
        "diffusers[torch]"


FROM build-torch AS build-dcnv4
ARG FORCE_CUDA=1
ARG TORCH_CUDA_ARCH_LIST="5.2 6.1 7.0 7.2 7.5 8.0 8.6 8.9 9.0"
RUN --mount=type=cache,target=/opt/ccache \
    git clone --depth=1 https://github.com/Kayaba-Akihiko/DCNv4.git /opt/dcnv4 && \
    cd /opt/dcnv4/DCNv4_op && \
    python setup.py bdist_wheel --dist-dir /opt/wheelhouse && \
    rm -rf /opt/dcnv4


FROM build-torch AS build-mamba
ARG FORCE_CUDA=1
ARG TORCH_CUDA_ARCH_LIST="5.2 6.1 7.0 7.2 7.5 8.0 8.6 8.9 9.0"
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/opt/ccache \
    python -m pip wheel \
        'git+https://github.com/state-spaces/mamba.git@v2.3.2' \
        'git+https://github.com/Dao-AILab/causal-conv1d.git@v1.6.2' \
        --no-build-isolation \
        --no-deps \
        -w /opt/wheelhouse && \
    uv pip install /opt/wheelhouse/*.whl

RUN --mount=type=cache,target=/opt/ccache \
    git clone --depth=1 https://github.com/Kayaba-Akihiko/VMamba.git /opt/vmamba && \
    cd /opt/vmamba/kernels/selective_scan && \
    sed -i 's/cub::LaneId()/((int)(threadIdx.x \& 31))/g' csrc/selective_scan/reverse_scan.cuh && \
    sed -i 's/cub::CTA_SYNC();/__syncthreads();/g' csrc/selective_scan/reverse_scan.cuh && \
    python setup.py bdist_wheel --dist-dir /opt/wheelhouse && \
    rm -rf /opt/vmamba


FROM build-torch AS build-pycma
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/opt/ccache \
    python -m pip wheel \
        "git+https://github.com/CMA-ES/pycma.git@v4.1.0" \
        --no-build-isolation \
        -w /opt/wheelhouse


FROM build-torch AS build-mmcv
ENV MMCV_WITH_OPS=1
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/opt/ccache \
    python -m pip wheel \
        "git+https://github.com/open-mmlab/mmengine.git@v0.10.7" \
        "git+https://github.com/open-mmlab/mmcv.git@v2.1.0" \
        mmsegmentation==1.2.2 \
        'mmpretrain[multimodal]==1.2.0' \
        --no-build-isolation \
        --no-deps \
        -w /opt/wheelhouse


FROM build-torch AS build-install-python


RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        torchmetrics \
        kornia \
        git+https://github.com/openai/CLIP.git \
        lpips \
        clean-fid \
        peft \
        open_clip_torch \
        ml-collections \
        pybind11 \
        eigen \
        einops \
        wandb \
        scikit-learn \
        scikit-learn-intelex \
        scikit-image \
        connected-components-3d \
        shap \
        numba \
        pandas \
        "matplotlib<3.11" \
        seaborn \
        SciencePlots \
        plotly \
        openpyxl \
        xlsx2csv \
        python-calamine \
        fastexcel \
        python-pptx \
        opencv-python \
        opencv-contrib-python \
        PyOpenGL \
        PyOpenGL_accelerate \
        open3d \
        "imageio[pyav]" \
        vtk \
        SimpleITK \
        pydicom \
        metaimageio \
        pylibjpeg \
        "pylibjpeg[all]" \
        pyjpegls \
        cairosvg \
        pygifsicle \
        openai \
        trimesh \
        optuna \
        mesh2sdf \
        probreg \
        "meshio[all]" \
        pdf2image \
        "pypdf[crypto]" \
        uncertainty-toolbox \
        git+https://github.com/MIC-DKFZ/nnUNet.git@v2.8.0 \
        monai \
        "torchio[plot,csv]" \
        point-cloud-utils

RUN --mount=type=cache,target=/root/.cache/uv \
    case "$TARGETARCH" in \
      amd64) \
        uv pip install polars ;; \
      arm64) \
        uv pip install polars-lts-cpu ;; \
      *) \
        echo "Unsupported TARGETARCH=$TARGETARCH" >&2; exit 1 ;; \
    esac

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install xformers --index-url https://download.pytorch.org/whl/cu126 && \
    python -m xformers.info

RUN --mount=type=cache,target=/root/.cache/uv \
    case "$TARGETARCH" in \
      amd64) \
        uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu126torch2.11-cp312-cp312-linux_x86_64.whl" ;; \
      arm64) \
        uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.22/flash_attn-2.8.3+cu126torch2.11-cp312-cp312-linux_aarch64.whl" ;; \
      *) \
        echo "Unsupported TARGETARCH=$TARGETARCH" >&2; exit 1 ;; \
    esac

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        torch_geometric \
        pyg_lib \
        torch_scatter \
        torch_sparse \
        -f https://data.pyg.org/whl/torch-2.11.0+cu126.html

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install mambavision --no-deps --no-build-isolation

COPY --from=build-dcnv4 /opt/wheelhouse/*.whl /opt/wheelhouse/
COPY --from=build-mamba /opt/wheelhouse/*.whl /opt/wheelhouse/
COPY --from=build-pycma /opt/wheelhouse/*.whl /opt/wheelhouse/
COPY --from=build-mmcv /opt/wheelhouse/*.whl /opt/wheelhouse/

RUN find /opt/wheelhouse -name '*.whl' -print && \
    uv pip install /opt/wheelhouse/*.whl && \
    rm -rf /opt/wheelhouse


FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS release
LABEL authors="gu"

ARG DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:/opt/bcpd/bcpd-master:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    MESA_LOADER_DRIVER_OVERRIDE=swrast

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=build-install-python $UV_PYTHON_INSTALL_DIR $UV_PYTHON_INSTALL_DIR
COPY --from=build-install-python $VIRTUAL_ENV $VIRTUAL_ENV

COPY --from=akihikokayaba/bcpd:ubuntu24.04 /opt/bcpd /opt/bcpd

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        cm-super \
        dvipng \
        ffmpeg \
        fontconfig \
        fonts-ipafont-gothic \
        fonts-ipafont-mincho \
        fonts-noto-cjk \
        gcc \
        gifsicle \
        libegl1 \
        libegl-mesa0 \
        libgbm1 \
        libglib2.0-0 \
        libgl1 \
        libgl1-mesa-dri \
        libglx-mesa0 \
        libglu1-mesa \
        libgomp1 \
        libopenblas0-pthread \
        libosmesa6 \
        libsm6 \
        libunwind-15 \
        libxext6 \
        libxrender1 \
        libreoffice \
        libreoffice-impress \
        mesa-utils \
        mesa-va-drivers \
        mesa-vdpau-drivers \
        openssl \
        poppler-utils \
        texlive-fonts-recommended \
        texlive-latex-extra \
        xvfb \
        x11-utils \
        openssh-server \
        rsync \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    case "$(uname -m)" in \
      x86_64)  LIBDIR=/usr/lib/x86_64-linux-gnu ;; \
      aarch64) LIBDIR=/usr/lib/aarch64-linux-gnu ;; \
      *) echo "Unsupported arch: $(uname -m)" >&2; exit 1 ;; \
    esac; \
    mkdir -p /usr/lib/dri /etc/ld.so.conf.d; \
    ln -sf "$LIBDIR/dri/swrast_dri.so" /usr/lib/dri/swrast_dri.so; \
    echo "$LIBDIR" > /etc/ld.so.conf.d/gcc.conf; \
    ldconfig

RUN set -eux; \
    which python; \
    python --version; \
    uname -m; \
    python -c "import platform; print(platform.machine())"; \
    for m in \
      torch \
      torchvision \
      torch_geometric \
      numpy \
      torchio \
      lightning \
      accelerate \
      transformers \
      diffusers \
      open3d \
      xformers \
      cv2; do \
        echo "=== importing $m ==="; \
        python -c "import ${m}; print('${m}', getattr(${m}, '__version__', 'no __version__'))"; \
    done;

RUN set -eux; \
    python -c "import mmengine; print(mmengine.__version__)"; \
    python -c "import mmcv; print(mmcv.__version__)"; \
    python -c "from mmcv.ops import DeformConv2d; print('mmcv ops OK')"; \
    python -c "import mmseg; print(mmseg.__version__)"; \
    python -c "import mmpretrain; print(mmpretrain.__version__)"