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
def build(
    fits: str,
    out_path: str,
    mode: str,
    workers: int | None,
    resume: bool,
    storage_options: str | None,
) -> None:
    """Build a survey store from FITS images.

    FITS is a directory or bucket prefix of FITS files, one FITS file, or a
    text file listing locations one per line.

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
    """
    from survey2zarray.stores.survey_store.build import build_survey_store

    build_survey_store(
        fits,
        out_path,
        mode=mode,
        max_workers=workers,
        resume=resume,
        storage_options=parse_storage_options(storage_options),
    )


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
