# survey2zarray

`survey2zarray` is a data pipeline for building multi-survey radio and optical astronomy PyTorch datasets from FITS images and source catalogues.

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

## Optical images

Optical coadds are built the same way. `s2z survey build` reads each image's band from its header (`BAND`, `FILTER`, or a similar keyword) and writes one survey store per band, so a directory of g, r, and z images built at `stores/legacy` becomes `stores/legacy_g`, `stores/legacy_r`, and `stores/legacy_z`. Where the headers record no band, `--band-pattern` reads it from the file names instead, for example `--band-pattern '-image-(?P<band>[a-z])\.'`.

A few further points apply to optical data:

- `--blank-value` names a pixel value that marks missing data, such as `0` for Legacy Surveys coadds. Those pixels read as NaN, so a cutout touching a coverage gap is not served.
- `--resolution` gives a point spread function FWHM in arcsec to images whose headers record no beam. Duplicate detection needs it to decide which sources a survey resolves.
- Tile-compressed `.fits.fz` files are read directly. They need copy mode, since their compressed pixels cannot be addressed as byte ranges.
- Only undistorted coadds are supported. Images with SIP or TPV distortion are refused when their headers are read.

## Store versions

Every store records the schema version of its on-disk layout (`schema_version`) and the `survey2zarray` version that wrote it (`created_with`). The schema version changes only when the layout does, so most releases read and write the same stores. A store with no recorded version counts as version 1.

A release opens stores written in any older schema. To append to such a store, upgrade it in place first:

```bash
s2z survey upgrade path/to/survey_store
s2z dataset upgrade path/to/dataset_store
```

An upgrade is recorded like any other edit, so one that fails or is interrupted leaves the store at its previous version. Do not upgrade a store while a training job is reading it. An older release refuses a store written in a newer schema, and its error names the release to install.

| Schema version | Survey store | Dataset store | Introduced in |
| --- | --- | --- | --- |
| 1 | Initial layout | Initial layout | 0.1.0.dev0 |
| 2 | Resolution arrays renamed from `bmaj`, `bmin`, and `bpa`; band and zeropoint added | As for survey stores, with a band per sub-store | 0.1.0.dev2 |
