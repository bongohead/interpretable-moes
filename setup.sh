#!/bin/bash

# Install necessary packages
pip install -U torch==2.4.1
pip install -U jupyter lab
pip install -U wandb
pip install -U bitsandbytes
pip install -U git+https://github.com/huggingface/transformers.git
pip install -U accelerate
pip install plotly.express
pip install scikit-learn
pip install -U flash-attn --no-build-isolation
pip install pyyaml
pip install pyarrow
pip install termcolor
pip install pandas
pip install tqdm
pip install sqlalchemy
pip install python-dotenv
pip install aiohttp
pip install asyncio

echo "All packages have been installed successfully."

# See https://github.com/pytorch/pytorch/issues/111469
# unset LD_LIBRARY_PATH