"""Load configuration from a mapping or a YAML file."""

import os
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Read a YAML file.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    result
        Parsed YAML mapping.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(input_config: str | Path | dict) -> dict:
    """Return a configuration mapping, reading from file if needed.

    Parameters
    ----------
    input_config
        A mapping to use as-is, or the path of a YAML file to read.

    Returns
    -------
    result
        Configuration mapping.

    Raises
    ------
    FileNotFoundError
        Where the path does not exist.
    TypeError
        Where the argument is of an unsupported type.
    """
    if isinstance(input_config, (str, Path)):
        if not os.path.exists(input_config):
            raise FileNotFoundError(
                f"input_config must name a file that exists, got {input_config}."
            )
        output_config = load_yaml(input_config)
    elif isinstance(input_config, dict):
        output_config = input_config
    else:
        raise TypeError(
            f"Expected str, Path, or dict for input_config, but got "
            f"{type(input_config).__name__}."
        )
    return output_config
