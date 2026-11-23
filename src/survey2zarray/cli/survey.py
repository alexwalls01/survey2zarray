"""Provide survey store commands."""

import json
import os

import click

STORAGE_OPTIONS_HELP = (
    "Filesystem options keyed by protocol, given as JSON or as the path to a "
    'JSON file, such as \'{"s3": {"anon": true}}\'. Needed where an archive '
    "will not serve anonymously, or wants a region or an endpoint."
)


def parse_storage_options(value: str | None) -> dict | None:
    """Read storage options given as JSON, or as the path to a JSON file.

    Parameters
    ----------
    value
        What was passed on the command line.

    Returns
    -------
    result
        Options keyed by storage protocol.

    Raises
    ------
    click.BadParameter
        Where it is not JSON, or does not describe options keyed by protocol.
    """
    if not value:
        return None
    text = value
    if os.path.isfile(value):
        with open(value) as handle:
            text = handle.read()
    try:
        options = json.loads(text)
    except json.JSONDecodeError as error:
        raise click.BadParameter(
            f"Expected JSON or a readable JSON file, but got {error}."
        ) from error
    if not isinstance(options, dict) or not all(
        isinstance(entry, dict) for entry in options.values()
    ):
        raise click.BadParameter(
            "Expected options keyed by protocol, for example "
            '\'{"s3": {"anon": true}}\'.'
        )
    return options


@click.group()
def survey() -> None:
    """Expose survey store commands."""


@survey.command("build")
@click.argument("fits")
@click.argument("out_path", type=click.Path(file_okay=False, writable=True))
@click.option(
    "--mode",
    type=click.Choice(["copy", "reference"]),
    default="copy",
    show_default=True,
    help="With copy the pixels are written into the store. With reference "
    "the files are mapped by byte range and read where they are.",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Headers to read at once. Defaults to a few for files on a server, "
    "one for files on disk.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume an interrupted build, keeping the headers it already read.",
)
@click.option("--storage-options", default=None, help=STORAGE_OPTIONS_HELP)
@click.option(
    "--band-pattern",
    default=None,
    help="Regular expression with a group named 'band', searched in the file name "
    "of each image whose header records no band, such as '-image-(?P<band>[a-z])'.",
)
@click.option(
    "--blank-value",
    type=float,
    default=None,
    help="Pixel value marking missing data, read as NaN, such as 0 for Legacy "
    "Surveys coadds.",
)
@click.option(
    "--resolution",
    type=float,
    default=None,
    help="Beam or point spread function FWHM in arcsec, for images whose headers "
    "record none.",
)
def build(
    fits: str,
    out_path: str,
    mode: str,
    workers: int | None,
    resume: bool,
    storage_options: str | None,
    band_pattern: str | None,
    blank_value: float | None,
    resolution: float | None,
) -> None:
    """Build a survey store from FITS images, one store per band.

    FITS is a directory or bucket prefix of FITS files, one FITS file, or a
    text file listing locations one per line. Images in several bands make one
    store each, named OUT_PATH_<band>.

    Parameters
    ----------
    fits
        FITS input specification.
    out_path
        Destination store path.
    mode
        Pixel access mode.
    workers
        Header reader count.
    resume
        Whether to resume an interrupted build.
    storage_options
        JSON filesystem options.
    band_pattern
        File-name pattern for bands.
    blank_value
        Pixel value marking missing data.
    resolution
        FWHM in arcsec for images whose headers record none.
    """
    from survey2zarray.stores.survey_store.build import build_survey_store

    stores = build_survey_store(
        fits,
        out_path,
        mode=mode,
        max_workers=workers,
        resume=resume,
        storage_options=parse_storage_options(storage_options),
        band_pattern=band_pattern,
        blank_value=blank_value,
        resolution=resolution,
    )
    click.echo(f"Wrote {', '.join(stores)}.")


@survey.command("append")
@click.argument("store_path", type=click.Path(exists=True, file_okay=False))
@click.argument("fits")
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of headers to read at once, as for build.",
)
@click.option("--storage-options", default=None, help=STORAGE_OPTIONS_HELP)
def append(
    store_path: str, fits: str, workers: int | None, storage_options: str | None
) -> None:
    """Append images to an existing survey store.

    Parameters
    ----------
    store_path
        Survey store path.
    fits
        FITS input specification.
    workers
        Header reader count.
    storage_options
        JSON filesystem options.
    """
    from survey2zarray.stores.survey_store.edit import append_tiles

    append_tiles(
        store_path,
        fits,
        max_workers=workers,
        storage_options=parse_storage_options(storage_options),
    )


@survey.command("retry")
@click.argument("store_path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of headers to read at once, as for build.",
)
@click.option("--storage-options", default=None, help=STORAGE_OPTIONS_HELP)
def retry(store_path: str, workers: int | None, storage_options: str | None) -> None:
    """Retry unreadable files recorded by a survey store build.

    Parameters
    ----------
    store_path
        Survey store path.
    workers
        Header reader count.
    storage_options
        JSON filesystem options.
    """
    from survey2zarray.stores.survey_store.edit import retry_failed_tiles

    retry_failed_tiles(
        store_path,
        max_workers=workers,
        storage_options=parse_storage_options(storage_options),
    )


@survey.command("upgrade")
@click.argument("store_path", type=click.Path(exists=True, file_okay=False))
def upgrade(store_path: str) -> None:
    """Upgrade a survey store to the current schema version.

    Parameters
    ----------
    store_path
        Survey store path.
    """
    from survey2zarray.stores.migrations import upgrade_survey_store

    upgrade_survey_store(store_path)
