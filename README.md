# Research Development Setup
1. Acquire a server with a CUDA-compatible GPU on Runpod.
2. Clone this repo: `git clone https://github.com/bongohead/interpretable-moes`
3. Add credentials to the repo if you don't have git credentials set already.
    ```
    git config user.email "{some_email_connected_to_your_github@email.com}"
    git config user.name {your_name}
    ```
4. To install necessary packages, run `sh runpod_setup.sh`.
5. Run `echo "/workspace/interpretable-moes" > $(python3 -c "import site; print(site.getsitepackages()[0])")/add_path.pth`, replacing `/workspace/interpretable-moes` with whatever absolute path you're using. This will add this directory to your Python's system paths so that module paths can be loaded with reference to the top-level folder path (e.g. so you can import from `helpers`).