# These are APT updates needed for use on Runpod
apt update -y && apt upgrade -y && apt install -y nano &&
pip install jupyterlab ipywidgets jupyterlab-widgets --upgrade &&\
sh /workspace/projects/interpretable-moes/install_packages.sh