# Research Development Setup
1. Acquire a server with a CUDA-compatible GPU on Runpod.
2. Clone this repo: `git clone https://github.com/bongohead/interpretable-moes`
3. Add credentials to the repo if you don't have git credentials set already.
    ```
    git config user.email "{some_email_connected_to_your_github@email.com}"
    git config user.name {your_name}
    ```
4. (If connecting via remote SSH e.g. on VS Code) Create a new venv `python3 -m venv .venv` and activate it, `source .venv/bin/activate` (**Note: skip the step if you're connecting via the Runpod-provided Jupyterlab port, since that one must utilize the base venv**)
5. To install necessary packages, run `sh runpod_setup.sh`.