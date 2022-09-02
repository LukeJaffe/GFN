# Gallery Filter Network for Person Search
This repository implements the models described in the paper, "Gallery Filter Network for Person Search".
The Object Search Research (OSR) package implements data prep, training, and inference for the
CUHK-SYSU and PRW datasets. The package is easily extensible to other datasets.

## Installation
The OSR package can be installed with docker or conda.
We provide example install instructions below, so the user can use
the commands in setup.py "out of the box". 

### docker
```
host$ docker build --no-cache -t osr:v1.0.0 -f Dockerfile .

host$ docker run -it --rm \
        --ulimit core=0 \
        --name=osr_$(date +%F_%H-%M-%S) \
        --runtime=nvidia \
        --net=host \
        -v /dev/shm:/dev/shm \
        -v <PRW_PATH>:/datasets/prw \
        -v <CUHK_PATH>:/datasets/cuhk \
        -v $(pwd)/weights:/weights/hub \
        -v $(pwd):/home/username \
        -w /home/username \
        osr:v1.0.0 bash -c \
                "chown -R $(id -u):$(id -g) /home/username;\
                 groupadd -g $(id -g) groupname;\
                 useradd -u $(id -u) -g $(id -g) -d /home/username username;\
                 su username -s /bin/bash;"

container$ export PATH=${PATH}:/opt/conda/bin
```

You can also re-install in the container with:
```
container$ python3 setup.py install --user
```

### conda
```
(base)$ conda env create -f conda.yaml

(base)$ conda activate osr

(osr)$ python3 setup.py install --user
```

## Data Download
Optionally install gdown python package for easy download of the datasets from google drive.
```
pip install --user gdown
```

### [PRW](https://github.com/liangzheng06/PRW-baseline)
```
cd $DATASET_DIR
gdown https://drive.google.com/uc?id=0B6tjyrV1YrHeYnlhNnhEYTh5MUU
unzip PRW-v16.04.20.zip -d prw
```

### [CUHK-SYSU](https://github.com/ShuangLI59/person_search)
```
cd $DATASET_DIR 
gdown https://drive.google.com/uc?id=1z3LsFrJTUeEX3-XjSEJMOBrslxD2T5af 
tar -xzvf cuhk_sysu.tar.gz -C cuhk
```

## Data Prep
After docker or conda installation of the package above, simply run:
```
osr_prep_cuhk --dataset_dir ${DATASET_DIR}/cuhk
osr_prep_prw --dataset_dir ${DATASET_DIR}/prw
```

## Config
For training and inference, we use .yaml files for the config format, with examples in the ./configs dir.
Config files inherit from ./configs/default.yaml, which has all possible parameters, with documentation.

To train or test, make sure to first modify the dataset_dir in the target config .yaml.

We include config files for all the experiments in the main paper:

    - baseline model
    - final model
    - augmentation ablation
    - crop size ablation
    - GFN objective ablation
    
Some configs group params together for easy running with ray tune grid_search. Additional config files, e.g., from supplementary experiments, are available upon request.

## Training
To train the final models:
```
osr_run --trial_config=./configs/cuhk_train_final.yaml
osr_run --trial_config=./configs/prw_train_final.yaml
```

## Inference
Trained model checkpoints are available upon request, but are too large to come with this export. To test, you will
need to run the training script, then modify the checkpoint path in the test .yaml files to the resulting training checkpoint.

To test the final models:
```
osr_run --trial_config=./configs/cuhk_test_final.yaml
osr_run --trial_config=./configs/prw_test_final.yaml
```

## Acknowledgment
Thanks to the authors of the following repos for their code, which was integral in this project:
- [SeqNet](https://github.com/serend1p1ty/SeqNet)
- [torchvision](https://github.com/pytorch/vision)
- [albumentations](https://github.com/albumentations-team/albumentations)
- [pytorch_metric_learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
