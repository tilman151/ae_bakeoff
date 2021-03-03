# The Great Autoencoder Bake Off

The companion repository to a post on [my blog](https://krokotsch.eu).
It contains all you need to reproduce the results.

## Features

Currently featured autoencoders:

* Shallow AE
* Deep (vanilla) AE
* Stacked AE
* Sparse AE
* Denoising AE
* VAE
* beta-VAE
* vq-VAE

They are evaluated on for the following tasks:

* Reconstruction quality
* Quality of decoded samples from the latent space (if possible)
* Quality of latent space interpolation
* Structure of the latent space visualized with [UMAP](https://github.com/lmcinnes/umap)
* ROC curve for anomaly detection with the reconstruction error
* Classification accuracy of a linear layer fitted on the autoencoder's features

Currently available datasets are:

* MNIST
* Fashion-MNIST (FMNIST)
* Kuzushiji-MNIST (KMNIST)

## Installation

Clone the repository and create a new conda environment with:

```shell
conda create -n ae_bakeoff python=3.7
conda activate ae_bakeoff
conda install --file requirements.txt -c pytorch -c conda-forge
```

Verify the installation by running the tests:

```shell
cd ./tests
export PYTHONPATH="../src"
python -m unittest
```

## Usage

To one-click reproduce the results for a dataset, call:

```shell
cd ./src
python reproduce.py --dataset <dataset> --batch_size 256 [--gpu]
```

If you want to run any specific experiment, call:

```shell
python run.py <autoencoder_type> --dataset <dataset> --batch_size 256 [--gpu] [--anomaly]
```

All experiments are recorded in the dicrectory `./logs/<dataset>`.
