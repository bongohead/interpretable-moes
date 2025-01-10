# These are APT updates needed for use on Runpod
apt update -y && apt upgrade -y && apt install -y nano &&
pip install jupyterlab ipywidgets jupyterlab-widgets --upgrade &&\
python3.11 -m venv .venv && sh install_packages.sh && source .venv/bin/activate && sh install_packages.sh