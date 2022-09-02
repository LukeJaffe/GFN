# Latest PyTorch image as of 07/08/2022
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Apt packages
RUN apt-get update
RUN apt-get install -y \
    vim=2:8.0.1453-1ubuntu1.8 \
    gcc=4:7.4.0-1ubuntu2.3 \
    wget=1.19.4-1ubuntu2.2 \
    rsync=3.1.2-2.1ubuntu1 \
    grsync=1.2.6-1

# Python packages
RUN pip install pytorch_metric_learning==1.5.1
RUN pip install torchvision==0.13.0
RUN pip install pycocotools==2.0.4
RUN pip install ray[tune]==1.13.0
RUN pip install albumentations==1.2.1
RUN pip install notebook==6.4.12
RUN pip install matplotlib==3.5.2
RUN pip install pandas==1.3.5
RUN pip install tensorflow==2.9.1
RUN pip install tensorboard==2.9.1

# Directory where torchvision model weights will be downloaded
ENV TORCH_HOME=/weights

# Install OSR
COPY ./ /opt/osr
WORKDIR /opt/osr
RUN python3 setup.py install
