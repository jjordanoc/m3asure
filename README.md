## Getting started
Due to the double-blind anonymity policies regarding external links, we kindly ask reviewers to download dependencies
manually. 

Please clone the standard public repositories for the following third-party models and place them in this repository's
root directory:
- VGGT (Official Facebook Research release)
- MEt3R (Public implementation)
- AnyUp (Public implementation)
If AnyUp doesn't build, create a new `pyproject.toml` for it from scratch.

## Data preparation
First, prepare the data. Sample data are provided under the 200 MB limit. The full ~10GB dataset will be disclosed after
the review process.

## Probe training
Modify the configs/paths in `main()`
```
cd inference/m3asure_one

uv run extract_features.py --batch-size 4
uv run train.py
```

## Probe inference
A different seed checkpoint is provided for inference as `m3asure_latent_ckpt.pt`.
Modify the configs/paths in `main()`
```
cd inference/m3asure_one

uv run inf_3dgenbench.py
```

## Projector and baselines inference
For clarity, files and folder structure are consistent for all models. Running them is as described above, with the
correct data.

