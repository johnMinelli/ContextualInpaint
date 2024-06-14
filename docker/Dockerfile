FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
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

## install dependencies
## See https://pytorch.org/ for other options if you use a different version of CUDA
#RUN pip3 install --user tensorboard cmake onnx matplotlib requests numpy jinja2 cog # cmake from apt-get is too old
#RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#
#RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
## install detectron2
#RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
## set FORCE_CUDA because during `docker build` cuda is not accessible
#ENV FORCE_CUDA="1"
## This will by default build detectron2 for all common cuda architectures and take a lot more time,
## because inside `docker build`, there is no way to tell which architecture will be used.
#ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
#ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
#
#RUN pip install --user -e detectron2_repo
#
## Set a fixed model cache directory.
#ENV FVCORE_CACHE="/tmp"
#
#RUN git clone --branch v1.1.3 https://github.com/haotian-liu/LLaVA.git LLaVA_repo
#RUN python -m pip install --upgrade pip
#RUN pip install -e LLaVA_repo
#RUN pip install git+https://github.com/facebookresearch/segment-anything.git

RUN git clone https://github.com/johnMinelli/ContextualInpaint.git
RUN pip install -r ContextualInpaint/requirements.txt

WORKDIR /home/appuser/ContextualInpaint/preprocess

# python preprocess.py --action mask --input_path /mounted_input --output_path /mounted_output  
# python preprocess.py --action pose --input_path /mounted_input --output_path /mounted_output
# python preprocess.py --action prompt --input_path /mounted_input --output_path /mounted_output

# cd ContextualInpaint/
# pip install -r requirements.txt
# python diffuser_train.py --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-inpainting --output_dir=../<model_store_folder>/ --train_data_dir ../<data_root_folder>/ --num_train_epochs 11 --sd_unlock 6 --checkpointing_steps 3000 --gradient_accumulation_steps 4 --train_batch_size 4
