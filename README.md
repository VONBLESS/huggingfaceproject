# HuggingFaceProject

Personal experiments for image/video generation with Hugging Face `diffusers`, plus small finance/serial scripts.

## What is in this repo

- `sfw.py`, `sfw2.py`, `run models.py`, `flux1.py`, `flux1schennel.py`, `lumina2.py`: text-to-image scripts
- `video.py`, `video2.py`: text-to-video scripts
- `financesender.py`, `send.py`: serial + finance utilities

## Prerequisites

- Windows 10/11 (project is currently Windows-oriented)
- Python 3.10+ (3.10 or 3.11 recommended)
- Git
- NVIDIA GPU + CUDA drivers for fast inference (CPU may be too slow for most scripts)

## Install from scratch

1. Clone the repo:

```powershell
git clone <YOUR_REPO_URL>
cd HuggingFaceProject
```

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

4. Install PyTorch for your CUDA version (pick one command from pytorch.org):

```powershell
# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

5. Install project dependencies:

```powershell
pip install -r requirements.txt
```

6. (Optional) Login for gated/private models:

```powershell
huggingface-cli login
```

## Run scripts

From the repo root with `.venv` activated:

```powershell
python .\sfw.py
python .\sfw2.py
python .\video2.py
python .\financesender.py
```

Generated media files are written in the project root unless changed in each script.

## Publish this repo to GitHub

If this local repo is not connected yet:

1. Create a new empty GitHub repo (for example `HuggingFaceProject`).
2. Add remote and push:

```powershell
git remote add origin https://github.com/<YOUR_USERNAME>/HuggingFaceProject.git
git add .
git commit -m "Initial project setup with install/run docs"
git branch -M main
git push -u origin main
```

If you already have a remote, use:

```powershell
git add .
git commit -m "Update docs and project setup"
git push
```

## Notes

- Most generation scripts assume CUDA (`.to("cuda")`).
- Some model names may require acceptance of model terms on Hugging Face.
- Keep large generated files out of git unless you intentionally want LFS.