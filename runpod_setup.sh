#!/bin/bash
# These are APT updates needed for use on Runpod

# Update & upgrade system packages
apt update -y \
&& apt upgrade -y

# Install nano
apt install -y nano

# Upgrade/install Jupyter Lab and widgets
pip install --upgrade jupyterlab ipywidgets jupyterlab-widgets

# Install Python 3.12
apt install -y python3.12

# Set Python 3.12 as the default python
update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install python3.12-venv
apt install -y python3.12-venv

# Ensure pip is up to date for Python
python -m ensurepip --upgrade

# Install packages
sh /workspace/projects/interpretable-moes/install_packages.sh