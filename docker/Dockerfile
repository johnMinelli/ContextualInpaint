FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y apt-utils && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils python3-opencv ca-certificates git wget sudo ninja-build curl && \
    ln -sv /usr/bin/python3.10 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"

# cuda 11.7
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb && \
    sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb && \
    sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    sudo apt-get update && \
    sudo apt-get -y install cuda-toolkit-11-7

# dependencies for detectron
RUN pip install -U tensorboard cmake onnx matplotlib requests numpy jinja2 cog
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install -U fvcore setuptools==69.5.1
ENV CUDA_HOME=/usr/local/cuda-11.3
ENV PATH=/usr/local/cuda-11.3/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

# Detectron2
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
RUN git clone https://www.github.com/facebookresearch/detectron2 detectron2_repo
RUN pip install --user -e detectron2_repo

# LLaVA
RUN git clone --branch v1.1.3 https://github.com/haotian-liu/LLaVA.git LLaVA_repo
RUN python -m pip install --upgrade pip
RUN pip install -e LLaVA_repo
RUN pip install git+https://github.com/facebookresearch/segment-anything.git


ENV CUDA_HOME=/usr/local/cuda-11.7
ENV PATH=/usr/local/cuda-11.7/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# GroundingDino
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
RUN python -m pip install --upgrade pip && \
    pip install wheel && \
    cd  GroundingDINO && \
    pip install -r requirements.txt && \
    pip install -e .
RUN mkdir weights && \
    cd weights && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# GroundedSAM
RUN pip install pillow==9.4
RUN git clone https://github.com/luca-medeiros/lang-segment-anything
RUN cd lang-segment-anything && sed -i '/Pillow = "9.3.0"/d' pyproject.toml && pip install -e .

# project code
RUN git clone https://github.com/johnMinelli/ContextualInpaint.git
RUN pip install -r ContextualInpaint/requirements.txt
RUN pip install huggingface-hub==0.23.3
RUN pip install transformers==4.34.0 lycoris_lora==2.2.0.post3
RUN pip install diffusers==0.26.0
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

WORKDIR /home/appuser/ContextualInpaint/preprocess

# python preprocess.py --action mask --input_path /mounted_input --output_path /mounted_output  
# python preprocess.py --action pose --input_path /mounted_input --output_path /mounted_output
# python preprocess.py --action prompt --input_path /mounted_input --output_path /mounted_output

# cd ContextualInpaint/
# pip install -r requirements.txt
# python diffuser_train.py --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-inpainting --output_dir=../<model_store_folder>/ --train_data_dir ../<data_root_folder>/ --num_train_epochs 11 --sd_unlock 6 --checkpointing_steps 3000 --gradient_accumulation_steps 4 --train_batch_size 4
