#!/bin/bash

# Install necessary packages
pip install torch==2.5.1
pip install bitsandbytes==0.45.2
pip install transformers==4.48.3
pip install accelerate==1.3.0
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install jupyter lab
pip install plotly.express
pip install wandb
pip install pyyaml
pip install pyarrow
pip install termcolor
pip install pandas
pip install tqdm
pip install sqlalchemy
pip install python-dotenv
pip install datasets

echo "All packages have been installed successfully."

# See https://github.com/pytorch/pytorch/issues/111469
# unset LD_LIBRARY_PATH