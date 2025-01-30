#!/bin/bash

# Install necessary packages
pip install torch==2.4.1
pip install jupyter lab
pip install wandb
pip install bitsandbytes
pip install git+https://github.com/huggingface/transformers.git
pip install accelerate
pip install plotly.express
pip install scikit-learn
pip install flash-attn --no-build-isolation
pip install pyyaml
pip install pyarrow
pip install termcolor
pip install pandas
pip install tqdm
pip install sqlalchemy
pip install python-dotenv
pip install aiohttp
pip install asyncio
pip install datasets

echo "All packages have been installed successfully."

# See https://github.com/pytorch/pytorch/issues/111469
# unset LD_LIBRARY_PATH