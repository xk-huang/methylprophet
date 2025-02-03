# Env

## .env

Create an `./.env` file to use wandb to log our experiments:

```bash
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=methylprophet
WANDB_MODE=online  # online, offline, disabled; ref: https://docs.wandb.ai/ref/python/init/
WANDB_ENTITY=your_user_name
```

If you do not want to use wandb, use the setting below in `./.env`:
```bash
WANDB_MODE=disabled
```

## Conda

Install mini conda into `BASE_DIR`

```shell
BASE_DIR=/path/to/your/target/dir

cd $BASE_DIR/misc
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -p $BASE_DIR/misc/miniconda3

# source ~/.bashrc
source activate base
rm Miniconda3-latest-Linux-x86_64.sh
```

<details>
<summary>If conda is not init in .rc files</summary>
The conda bashrc config:
Use `$BASE_DIR/misc/miniconda/bin/conda init`
Remember to replace your `$BASE_DIR`.

```shell
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('$BASE_DIR/misc/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$BASE_DIR/misc/miniconda/etc/profile.d/conda.sh" ]; then
        . "$BASE_DIR/misc/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="$BASE_DIR/misc/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```
</details>

Follow `nvcr.io/nvidia/pytorch:24.06-py3` in https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-06.html to install our packages.

We assume the cuda is 12.4.

```bash
# Install python
conda create -n xiaoke-methylformer python=3.10
source activate
conda activate xiaoke-methylformer
which python
# should be `/insomnia001/depts/5sigma/users/xh2689/xiaoke/misc/miniconda/envs/xiaoke-methylformer/bin/python`

# Install pytorch 2.4.0
# https://pytorch.org/get-started/locally/
# We use cuda 12.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
# conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

pip install tensorboard scikit-learn
pip install -r requirements.txt
pip install -e .
```

Install flash-attn

Check CUDA is added to your system PATH by `nvcc -V`.

Check or add CUDA to PATH:

```shell
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cat >> ~/.zshrc << EOF
export PATH=\$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64
EOF
```

Need `flash-attn`'s version > 2

```shell
pip install packaging ninja
pip uninstall -y ninja && pip install ninja
pip install flash-attn --no-build-isolation
```


## Docker

Pull 
```shell
docker pull xkhuang2022/methylformer:241115
docker tag xkhuang2022/methylformer:241115 methylformer:241115

docker tag methylformer:241115 xkhuang2022/methylformer:241115
docker push xkhuang2022/methylformer:241115
```

Build

```shell
IMAGE_NAME="methylformer:241115"

docker build -t $IMAGE_NAME -f docker/dockerfile .
```

Run container

```shell
BASE_DIR=$(pwd)
IMAGE_NAME="methylformer:241115"
CONTAINER_NAME="241115-methylformer-xiaoke"

docker run -td \
--gpus all \
--ipc=host \
--network host \
--ulimit memlock=-1 --ulimit stack=67108864 \
--user $(id -u ${USER}):$(id -g ${USER}) \
-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro -v $HOME:$HOME \
--group-add $(getent group docker | cut -d: -f3) --group-add $(getent group sudo | cut -d: -f3) \
-v $BASE_DIR:$BASE_DIR \
-w $(pwd) \
--name $CONTAINER_NAME \
$IMAGE_NAME

CONTAINER_NAME="241115-methylformer-xiaoke"
docker exec -it $CONTAINER_NAME bash
```

NOTE: do not mount `$HOME`. Since some packages are installed in `$HOME/.local` which may cause contradiction.

#### No local stored user info

Write your user info first.

```shell
mkdir -p misc
cat /etc/passwd >> misc/passwd
getent passwd $USER >> misc/passwd

BASE_DIR=$(pwd)
IMAGE_NAME="methylformer:241115"
CONTAINER_NAME="241115-methylformer-xiaoke"

docker build -t $IMAGE_NAME -f docker/dockerfile .

docker run -td  \
--gpus all \
--ipc=host \
--network host \
--ulimit memlock=-1 --ulimit stack=67108864 \
--user $(id -u ${USER}):$(id -g ${USER}) \
--group-add $(getent group docker | cut -d: -f3) --group-add $(getent group sudo | cut -d: -f3) \
-v $(realpath misc/passwd):/etc/passwd:ro  \
-v $BASE_DIR:$BASE_DIR -v $HOME:$HOME \
-w $(pwd) \
--name $CONTAINER_NAME \
$IMAGE_NAME


CONTAINER_NAME="241115-methylformer-xiaoke"
docker exec -it $CONTAINER_NAME bash
```

### AWS Linux

```bash
docker run -td \
--gpus all \
--ipc=host \
--network host \
--ulimit memlock=-1 --ulimit stack=67108864 \
--user $(id -u ${USER}):$(id -g ${USER}) \
-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro \
-v $BASE_DIR:$BASE_DIR -v $HOME:$HOME \
-w $(pwd) \
--name $CONTAINER_NAME \
$IMAGE_NAME
```

## Apptainer

```shell
apptainer build data/methylformer.sif docker/apptainer.def

apptainer shell -B /blue:/blue data/methylformer.sif
```

NOTE: `cudf 24.4.0` is installed in nvcr pytorch_24.06. 
Installing requirement.txt leads to `cudf 24.4.0 requires pyarrow<15.0.0a0,>=14.0.1, but you have pyarrow 17.0.0 which is incompatible.`.
But I think it is OK.

