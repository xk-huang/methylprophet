# MODEL


## Model List

See `src/models/model_factory.py` and `src/configs/models/`.


## Add Model

Write model file in `src/models/`. First write `Config` class, then define `Model` class.

Update `src/models/model_factory.py`. Import `Config` and `Model` class, then update `MODEL_NAME_TO_MODEL_CLASS` and `MODEL_CONFIG_NAME_TO_MODEL_CONFIG_CLASS`.

Write config files in `src/configs/models/`. Change `model_class` and `model_config_class` according to `src/models/model_factory.py`. and update args according to `Config` class.


## RoPE Base

https://kexue.fm/archives/10122
"Base of RoPE Bounds Context Length"


RoPE Base in Large World Model

| Max Seq Len | 2^15 (32K) | 2^17 (131K) | 2^18 (262K) | 2^19 (524K) | 2^20 (1M)|
| RoPE theta | 1M | 10M | 10M | 25M | 50M |


Rope Base in Llama-3 
Max Seq len 2^13 (8192)
RoPE theta: 50M


Chr 1
seq len: 248,956,422, 249M 
theta: 50M * 256


Chr 21
seq len: 46,709,983, 46M
theta: 50M * 46

## ConvNext for DNA Sequence Encoding

"VQDNA: Unleashing the Power of Vector Quantization for Multi-Species Genomic Sequence Modeling", "3.3. Implementation Details"

> The encoder network for VQVAE and HRQ consists of a stem module and 6 residual blocks, i.e., N=6, and D=384.
> The stem projects the input data (one-hot encoded) to 256 dimensions by a 1D convolution layer with a kernel size of 5 and a stride of 1, followed by a LayerNorm and GELU activation.
> Each residual block contains a 1D depth-wise convolution layer (the kernel size of 7) and 2 full-connected layers to form the inverted bottleneck (expanding 4 times).

NOTE: Similar to Isotropic ConvNextV1 https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext_isotropic.py

"A ConvNet for 2020s" 
> Due to the redundancyinherent in natural images, a common stem cell will aggressively downsample the input images to an appropriate feature map size in both standard ConvNets and vision Transformers.

Thats why there is not downsampling in stem, as DNA sequence is not redundant.

## Add Data Preprocessing

1. Add a data preprocessor in `src/data`.
2. Import the class in `src/data/utils.py`, and add it into `DATA_PREPROCESS_TYPE` and `create_data_preprocess`
3. In `src/configs/data_preprocess.py`, add new class name and args.

