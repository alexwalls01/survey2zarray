"""Provide dataset store commands."""

import click


@click.group()
def dataset() -> None:
    """Expose dataset store commands."""


@dataset.command("build")
@click.argument(
    "survey_stores",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.argument("catalogue", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_path", type=click.Path(file_okay=False, writable=True))
@click.option(
    "--max-cutout-size",
    type=float,
    required=True,
    help="Largest square, in arcsec across, to serve at any rotation. Every "
    "later read asks for this or less, and the disc stored is sqrt(2) times "
    "this across.",
)
@click.option(
    "--shard-size",
    type=int,
    default=None,
    help="Rows per shard of the pixel arrays, which is what a batch aligns "
    "to. Must be a power of two. Defaults to as many rows as fill a 512 MB "
    "shard, derived per survey from its cutout size and its coverage.",
)
@click.option(
    "--bucket-by",
    "bucket_by",
    multiple=True,
    help="Catalogue column to sort and deal the sources by. Repeatable, most "
    "important first. Defaults to every column the catalogue holds, "
    "discrete columns ahead of continuous columns.",
)
@click.option(
    "--n-buckets",
    type=int,
    default=None,
    help="Buckets to deal the sorted catalogue into, enabling block-aligned "
    "batch sampling. Default depends on catalogue's "
    "size and stratum count.",
)
@click.option(
    "--bucket-seed",
    type=int,
    default=0,
    show_default=True,
    help="Seed for the tie-break within the sort and the shuffle within each bucket.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume an interrupted build, keeping the cutouts it already "
    "extracted. A finished store, or one built with different settings, "
    "is rebuilt regardless.",
)
def build(
    survey_stores: tuple[str, ...],
    catalogue: str,
    out_path: str,
    max_cutout_size: float,
    shard_size: int | None,
    bucket_by: tuple[str, ...],
    n_buckets: int | None,
    bucket_seed: int,
    resume: bool,
) -> None:
    """Build a dataset store from survey stores and a source catalogue.

    Parameters
    ----------
    survey_stores
        Input survey store paths.
    catalogue
        Source catalogue path.
    out_path
        Destination store path.
    max_cutout_size
        Maximum cutout diameter in arcsec.
    shard_size
        Shared shard row count.
    bucket_by
        Catalogue columns used for stratification.
    n_buckets
        Requested bucket count.
    bucket_seed
        Seed for bucket assignment.
    resume
        Whether to resume an interrupted build.
    """
    from survey2zarray.stores.dataset_store.build import build_dataset_store
    from survey2zarray.stores.dataset_store.layout import is_power_of_two

    if shard_size is not None and not is_power_of_two(shard_size):
        raise click.BadParameter(
            "Expected a power-of-two row count.", param_hint="--shard-size"
        )
    if n_buckets is not None and n_buckets < 1:
        raise click.BadParameter(
            "Expected at least one bucket.", param_hint="--n-buckets"
        )
    build_dataset_store(
        list(survey_stores),
        catalogue,
        max_cutout_size,
        out_path,
        resume=resume,
        shard_size=shard_size,
        bucket_by=list(bucket_by) or None,
        n_buckets=n_buckets,
        bucket_seed=bucket_seed,
    )


@dataset.command("append-sources")
@click.argument("store_path", type=click.Path(exists=True, file_okay=False))
@click.argument("catalogue", type=click.Path(exists=True, dir_okay=False))
def append_sources(store_path: str, catalogue: str) -> None:
    """Append sources to an existing dataset store.

    Parameters
    ----------
    store_path
        Dataset store path.
    catalogue
        Source catalogue path.
    """
    from survey2zarray.stores.dataset_store.edit import (
        append_sources as append_sources_to_store,
    )

    append_sources_to_store(store_path, catalogue)


@dataset.command("append-survey")
@click.argument("store_path", type=click.Path(exists=True, file_okay=False))
@click.argument("survey_store", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--name",
    default=None,
    help="Name for the new survey sub-store. Defaults to the survey store's "
    "directory name.",
)
def append_survey(store_path: str, survey_store: str, name: str | None) -> None:
    """Append a survey sub-store to an existing dataset store.

    Parameters
    ----------
    store_path
        Dataset store path.
    survey_store
        Survey store path.
    name
        Name of the new sub-store.
    """
    from survey2zarray.stores.dataset_store.edit import (
        append_survey as append_survey_to_store,
    )

    append_survey_to_store(store_path, survey_store, survey_name=name)
