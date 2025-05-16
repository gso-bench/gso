# gso

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone --recursive https://github.com/r2e-project/pyperf.git
cd pyperf && uv venv && source .venv/bin/activate
uv sync
```

(Additional) Setup [HuggingFace](https://huggingface.co/docs/hub/en/security-tokens) token: 
```
export HF_TOKEN="huggingface_token"
```
