FROM haydnj/torch:latest

RUN uv pip install \
    "botorch>=0.11.3" \
    "gpytorch>=1.12" \
    "wandb>=0.17.7" \
    "torch>=2.4.0" \
    "torchvision>=0.19.0" \
    "torchaudio>=2.4.0" \
    "fire>=0.6.0" \
    "lightning>=2.4.0" \
    "selfies>=2.1.2" \
    "guacamol" \
    "ipython>=8.27.0" \
    "ipykernel>=6.29.5" \
    "polars>=1.6.0" \
    "pandas>=2.2.2" \
    "pyarrow>=17.0.0"

RUN uv pip install linear-operator==0.5.3