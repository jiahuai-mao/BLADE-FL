FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    ca-certificates \
    bzip2 \
    build-essential \
    bash \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh

WORKDIR /workspace

COPY environment.docker.yml /tmp/environment.docker.yml

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda install -n base -y conda-libmamba-solver && \
    conda config --set solver libmamba

RUN conda env create -f /tmp/environment.docker.yml --solver=libmamba

RUN conda clean -afy

# Auto-activate jh_env for interactive shells
RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate jh_env" >> /root/.bashrc

CMD ["bash"]
