# DNA sequence WGAN-GP (adapted)

This directory contains the DNAzyme-specific WGAN-GP code adapted from [Improved Training of Wasserstein GANs](https://github.com/igul222/improved_wgan_training).

**Included scripts:**

- `gan_language.py` — pretrain and fine-tune on DNA sequence datasets
- `generate_sequences.py` — batch generation from checkpoints
- `evaluate_checkpoints.py` — JSD-based checkpoint evaluation
- `language_helpers.py` — sequence encoding / n-gram utilities
- `tflib/` — TensorFlow helper library (required dependency)
- `run_pretrain.sh` — pretrain launcher

Image-model demo scripts (`gan_mnist.py`, `gan_cifar.py`, etc.) are excluded from this repository.

See the [project README](../README.md) for full reproduction instructions.
