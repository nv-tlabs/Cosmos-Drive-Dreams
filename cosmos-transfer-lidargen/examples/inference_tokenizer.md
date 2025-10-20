## Inference with tokenizer models

### Environment setup

Please refer to the Inference section of [INSTALL.md](/INSTALL.md#inference) for instructions on environment setup.

### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```
3. Accept the [Llama-Guard-3-8B terms](https://huggingface.co/meta-llama/Llama-Guard-3-8B)

4. Download the Cosmos Tokenize model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7):
```bash
python3 -m scripts.download_tokenizer_checkpoints --checkpoint_dir checkpoints
```

The downloaded files should be in the following structure:
```
checkpoints/
├── Cosmos-Tokenize1-CV8x8x8-720p
├── Cosmos-Tokenize1-DV8x16x16-720p
├── Cosmos-Tokenize1-CI8x8-360p
├── Cosmos-Tokenize1-CI16x16-360p
├── Cosmos-Tokenize1-CV4x8x8-360p
├── Cosmos-Tokenize1-DI8x8-360p
├── Cosmos-Tokenize1-DI16x16-360p
└── Cosmos-Tokenize1-DV4x8x8-360p
```

Under the checkpoint repository `checkpoints/<model-name>`, we provide the encoder, decoder, the full autoencoder in TorchScript (PyTorch JIT mode) and the native PyTorch checkpoints. For instance for `Cosmos-Tokenize1-CV8x8x8-720p` model:
```bash
├── checkpoints/
│   ├── Cosmos-Tokenize1-CV8x8x8-720p/
│   │   ├── encoder.jit
│   │   ├── decoder.jit
│   │   ├── autoencoder.jit
│   │   ├── model.pt
```


### Examples
Cosmos Tokenizer contains tokenizers for both images and videos. The video tokenizers are temporally causal, as such they can operate as image tokenizers. For each media type, we provide tokenizers that operate both on continuous and discrete latent spaces.

The following table summarizes the nomenclature used for naming the various tokenizers, categorized by media type (images and videos) and latent space type (continuous and discrete):

|                   | Continuous ( C )    | Discrete ( D )      |
| ------------------|---------------------|---------------------|
| **Images ( I )**        | Cosmos-Tokenize-CI      | Cosmos-Tokenize-DI      |
| **Videos ( V )**        | Cosmos-Tokenize-CV      | Cosmos-Tokenize-DV      |

There are two modes of running inference. Under each mode, we provide example commands for encoding and decoding. We use the example images and videos provided in `cosmos_predict1/tokenizer/test_data/` to demo the inference scripts.
1. JIT/TorchScript
2. PyTorch

### Example 1. JIT / TorchScript Inference

#### Autoencoding images
Accepts an input image filepath, and outputs a reconstruction of the image obtained by decoding the encoded latents. For more help, type
```bash
python3 -m cosmos_predict1.tokenizer.inference.image_cli --help
```
```bash
# Autoencoding images using `Cosmos-Tokenize1-CI8x8-360p`.
model_name="Cosmos-Tokenize1-CI8x8-360p"
python3 -m cosmos_predict1.tokenizer.inference.image_cli \
    --image_pattern 'cosmos_predict1/tokenizer/test_data/image.png' \
    --checkpoint_enc checkpoints/${model_name}/encoder.jit \
    --checkpoint_dec checkpoints/${model_name}/decoder.jit
```
If `--output_dir` is not specified, you can find the reconstructed image at `cosmos_predict1/tokenizer/test_data/reconstructions/image.png`.

#### Autoencoding Videos
Accepts an input video filepath, and outputs a reconstruction of the video obtained by decoding the encoded latents. For more help, type
```bash
python3 -m cosmos_predict1.tokenizer.inference.video_cli --help
```
```bash
# Autoencoding videos using `Cosmos-Tokenize1-DV4x8x8-360p`.
model_name="Cosmos-Tokenize1-DV4x8x8-360p"
python3 -m cosmos_predict1.tokenizer.inference.video_cli \
    --video_pattern 'cosmos_predict1/tokenizer/test_data/video.mp4' \
    --checkpoint_enc checkpoints/${model_name}/encoder.jit \
    --checkpoint_dec checkpoints/${model_name}/decoder.jit
```
If `--output_dir` is not specified, then you can find the reconstructed video at `cosmos_predict1/tokenizer/test_data/reconstructions/video.mp4`.

### Example 2. PyTorch Inference

To run the tokenizers in native PyTorch, append your commands with `--mode=torch`. In PyTorch mode, the model is constructed from the native network definition, which requires providing additional arguments to configure the model for instantiation.

For example, to instantiate a `DI` tokenizer with a spatial compression factor of 8, operating at 720p or higher resolution, append the following command line arguments:

- `--mode=torch`
- `--tokenizer_type=DI8x8-360p`

Note that the `--checkpoint_enc`, `--checkpoint_dec`, and `--checkpoint` should still refer to JIT files. The necessary `state_dict`s will be extracted from the loaded JIT models.

#### Autoencoding Images
```bash
# Autoencoding images using `Cosmos-Tokenize1-DI8x8-360p`.
model_name="Cosmos-Tokenize1-DI8x8-360p"
python3 -m cosmos_predict1.tokenizer.inference.image_cli \
    --image_pattern 'cosmos_predict1/tokenizer/test_data/*.png' \
    --mode=torch \
    --tokenizer_type=DI8x8-360p \
    --checkpoint_enc checkpoints/${model_name}/encoder.jit \
    --checkpoint_dec checkpoints/${model_name}/decoder.jit
```

#### Autoencoding Videos
To instantiate a `CV` tokenizer with a temporal factor of 8 and a spatial compression factor of 8, operating at 720p or higher resolution, append the following command line arguments:

- `--mode=torch`
- `--tokenizer_type=CV8x8x8-720p`

```bash
# Autoencoding videos using `Cosmos-Tokenize1-CV8x8x8-720p`.
model_name="Cosmos-Tokenize1-CV8x8x8-720p"
python3 -m cosmos_predict1.tokenizer.inference.video_cli \
    --video_pattern 'cosmos_predict1/tokenizer/test_data/*.mp4' \
    --mode=torch \
    --tokenizer_type=CV8x8x8-720p \
    --checkpoint_enc checkpoints/${model_name}/encoder.jit \
    --checkpoint_dec checkpoints/${model_name}/decoder.jit
```

Similarly, to instantiate a `Cosmos-Tokenize1-CV4x8x8-360p`, append the following command line arguments and the corresponding jit compiled ckpts:
- `--mode=torch`
- `--tokenizer_type=CV4x8x8-360p`

```bash
# Autoencoding videos using `Cosmos-Tokenize1-CV4x8x8-360p`.
model_name="Cosmos-Tokenize1-CV4x8x8-360p"
python3 -m cosmos_predict1.tokenizer.inference.video_cli \
    --video_pattern 'cosmos_predict1/tokenizer/test_data/*.mp4' \
    --mode=torch \
    --tokenizer_type=CV4x8x8-360p \
    --checkpoint_enc checkpoints/${model_name}/encoder.jit \
    --checkpoint_dec checkpoints/${model_name}/decoder.jit
```

### Example 3: Encoding Videos
You can use the following examples for encoding videos into:
- Continuous embeddings.
- Discrete integers.

#### Encoding into Continuous Latent Space

```python
import torch
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer

model_name = "Cosmos-Tokenize1-CV4x8x8-360p"
input_tensor = torch.rand(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)  # [B, C, T, H, W]
input_tensor = input_tensor * 2. - 1.  # Normalize to [-1..1]
encoder = CausalVideoTokenizer(checkpoint_enc=f'checkpoints/{model_name}/encoder.jit')
(latent,) = encoder.encode(input_tensor)
torch.testing.assert_close(latent.shape, (1, 16, 3, 64, 64))

# The input tensor can be reconstructed by the decoder as:
decoder = CausalVideoTokenizer(checkpoint_dec=f'checkpoints/{model_name}/decoder.jit')
reconstructed_tensor = decoder.decode(latent)
torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)
```
The `latent` will have the shape `(1, 16, 3, 64, 64)`, where the first of the three latents represents the first frame, and C=16 is the number of channels of the latent.

#### Encoding into Discrete Tokens
```python
import torch
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer

model_name = "Cosmos-Tokenize1-DV4x8x8-360p"
input_tensor = torch.rand(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)  # [B, C, T, H, W]
input_tensor = input_tensor * 2. - 1.  # Normalize to [-1..1]
encoder = CausalVideoTokenizer(checkpoint_enc=f'checkpoints/{model_name}/encoder.jit')
(indices, codes) = encoder.encode(input_tensor)
torch.testing.assert_close(indices.shape, (1, 3, 64, 64))
torch.testing.assert_close(codes.shape, (1, 6, 3, 64, 64))

# The input tensor can be reconstructed by the decoder as:
decoder = CausalVideoTokenizer(checkpoint_dec=f'checkpoints/{model_name}/decoder.jit')
reconstructed_tensor = decoder.decode(indices)
torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)
```
The `indices` will have the shape `(1, 3, 64, 64)` and contain integral values in the range `[1..64K]`, where the first of the three integral maps represents the first frame.
The `codes` will contain the pre-quantization continuous latent with shape `(1, 6, 3, 64, 64)`, where C=6 represents the number of FSQ levels.

## Web Demo

* Image Tokenization [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/tokenizer/notebook/Image_Tokenization.ipynb)
* Video Tokenization [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/tokenizer/notebook/Video_Tokenization.ipynb)
