# Latest PyTorch image as of 01/20/2023
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Apt packages
RUN apt-get update
RUN apt-get install -y \
    vim \
    gcc=4:7.4.0-1ubuntu2.3 \
    wget=1.19.4-1ubuntu2.2 \
    rsync=3.1.2-2.1ubuntu1 \
    grsync=1.2.6-1

# Python packages
RUN pip install pytorch_metric_learning==1.5.1
RUN pip install torchvision
RUN pip install pycocotools
RUN pip install ray[tune]==1.13.0
RUN pip install notebook==6.4.12
RUN pip install matplotlib==3.5.2
RUN pip install pandas
RUN pip install tensorflow==2.9.1
RUN pip install tensorboard==2.9.1
RUN pip install pydantic==1.10.13
RUN pip install albumentations==1.4.3
RUN pip install numpy==1.23.1

# More python packages
RUN conda install -c conda-forge pytorch-lightning lightning-bolts torchmetrics

# Directory where torchvision model weights will be downloaded
ENV TORCH_HOME=/weights
