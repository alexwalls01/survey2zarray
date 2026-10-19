"""Provide the command-line entry point."""

import click

from survey2zarray.cli.dataset import dataset
from survey2zarray.cli.survey import survey


@click.group()
def cli() -> None:
    """Expose the multi-view radio-astronomy dataset pipeline."""


cli.add_command(survey)
cli.add_command(dataset)

if __name__ == "__main__":
    cli()
