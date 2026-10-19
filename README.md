# survey2zarray

`survey2zarray` is a data pipeline for building multi-survey radio astronomy PyTorch datasets from FITS images and source catalogues.

The pipeline has 3 stages:

1. A **survey store** indexes one survey's tiles and metadata. It is built from a set of FITS files, which can be local or remote. In `copy` mode, the pixel data is written into the store alongside the metadata. In `reference` mode, only a map of where each block of pixels sits within the FITS files is written, and the pixels are read from those files on demand, so that a large survey does not need to be duplicated.

2. A **dataset store** is built using one or more survey stores and a catalogue of source coordinates; for each catalogue source, it holds one cutout per survey (if the source is covered by that survey) and its metadata.

3. A `torch.utils.data.Dataset` (`survey2zarray.dataset.S2ZDataset`) reads cutouts from a dataset store, applies preprocessing and augmentation specified by a configuration file, and returns spatially aligned views of the source from each survey that covers it.

Both survey stores and dataset stores are [zarr](https://zarr.dev) v3 hierarchies, with their metadata consolidated into the root so that opening a store is a single read.

## Installation

`survey2zarray` requires Python 3.13 or later.

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/alexwalls01/survey2zarray.git
cd survey2zarray
git checkout v0.1.0.dev0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For development, install the package in editable mode with its test and lint
dependencies:

```bash
python -m pip install -e ".[dev]"
```

Check the installed version and command-line interface:

```bash
python -c "from importlib.metadata import version; print(version('survey2zarray'))"
s2z --help
```
