"""Build a survey store by indexing FITS images, then writing pixels and metadata."""

import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from itertools import repeat
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import zarr
from astropy.io import fits
from tqdm import tqdm
from zarr.codecs import BloscCodec

from survey2zarray.stores.store import consolidate, create_store
from survey2zarray.stores.survey_store.fits import (
    FITSOpenError,
    check_compatibility,
    extract_metadata,
    extract_stokes_i,
    get_image_hdu,
    quiet_fits_warnings,
)
from survey2zarray.stores.survey_store.references import (
    COMPRESSED_SUFFIXES,
    REFERENCE_MAP_PATH,
    compression_suffix,
    is_remote,
    open_remote,
    options_for,
    storage_protocol,
    write_json_atomically,
)
from survey2zarray.utils.logging import get_logger
from survey2zarray.utils.parallel import polite_workers

logger = get_logger(__name__)

# Retries with increasing delays tolerate transient archive load failures.
HEADER_ATTEMPTS: int = 3
RETRY_PAUSE: float = 1.0

# Checkpoint completed header reads beside the target store.
HEADER_CHECKPOINT_SUFFIX: str = ".headers.jsonl"

# Cap independent header requests below REMOTE_CONCURRENCY to protect archives.
HEADER_WORKERS: int = 4

# Server wait time permits at least two header readers on small machines.
REMOTE_MINIMUM_WORKERS: int = 2

# Accept compressed suffixes because some archives omit a FITS suffix.
FITS_SUFFIXES: tuple[str, ...] = (".fits", ".fit", ".fts")

STRING_METADATA_KEYS: list[str] = [
    "ctype",
    "image_location",
    "telescope",
    "bunit",
    "radesys",
    "header",
]

# Exclude credential options because stored configuration travels with every copy.
SECRET_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "key",
        "secret",
        "token",
        "password",
        "username",
        "credential",
        "credentials",
        "client_secret",
        "session_token",
        "access_key",
        "secret_key",
        "account_key",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "sas_token",
        "connection_string",
    }
)


def persistable_options(storage_options: dict[str, dict] | None) -> dict[str, dict]:
    """Return the storage options that may be written into a store.

    What is kept is the part that says where and how to reach an archive (a
    region, an endpoint, ``anon``), which a reader cannot be expected to guess
    and which is not a secret. What is dropped is anything naming a credential.
    A reader must supply those itself, which is what `SurveyStore.open` takes
    `storage_options` for.

    Parameters
    ----------
    storage_options
        Options keyed by protocol, as `options_for` takes them.

    Returns
    -------
    result
        The same, with every credential removed.
    """
    kept: dict[str, dict] = {}
    dropped: list[str] = []
    for protocol, options in (storage_options or {}).items():
        safe = {
            key: value
            for key, value in options.items()
            if key.lower() not in SECRET_OPTION_KEYS
        }
        dropped.extend(
            f"{protocol}.{key}" for key in options if key.lower() in SECRET_OPTION_KEYS
        )
        if safe:
            kept[str(protocol)] = safe
    if dropped:
        logger.warning(
            f"Not recording {len(dropped)} credential option(s) in the store "
            f"({', '.join(sorted(dropped))}). A reader must supply them itself."
        )
    return kept


def is_fits_name(location: str) -> bool:
    """Return whether a location names a file a reader should be offered.

    A compressed file may have no FITS suffix at all (NVSS ships
    ``C0000M04.gz``), so anything compressed counts, and a file that cannot
    then be opened is skipped like any other unreadable file.

    Parameters
    ----------
    location
        Path or URL of a file.

    Returns
    -------
    result
        Whether the location has a supported FITS suffix.
    """
    name = urlparse(str(location)).path.rsplit("/", 1)[-1].lower()
    return name.endswith(FITS_SUFFIXES) or name.endswith(COMPRESSED_SUFFIXES)


def resolve_remote_fits_list(
    location: str,
    storage_options: dict[str, dict] | None = None,
) -> list[str]:
    """Resolve a remote FITS file, prefix, or location list.

    Use fsspec protocol listing for prefixes and text files for protocols that
    expose no directory listing.

    Parameters
    ----------
    location
        URL of a FITS file, of a prefix holding them, or of a text file listing
        them one per line.
    storage_options
        Filesystem options keyed by protocol, as `options_for` takes them.

    Returns
    -------
    result
        Locations in a stable order.

    Raises
    ------
    FileNotFoundError
        Where nothing is there, or a prefix holds no FITS files.
    """
    import fsspec

    protocol = storage_protocol(location)
    filesystem = fsspec.filesystem(protocol, **options_for(protocol, storage_options))

    if is_fits_name(location):
        return [str(location)]

    if filesystem.isdir(location):
        found = sorted(
            filesystem.unstrip_protocol(str(path))
            for path in filesystem.ls(location, detail=False)
            if is_fits_name(str(path))
        )
        if not found:
            raise FileNotFoundError(f"No FITS files found under {location}.")
        logger.info(f"Found {len(found)} FITS files under {location}.")
        warn_if_compressed(found)
        return found

    if not filesystem.exists(location):
        raise FileNotFoundError(f"{location} does not exist.")

    listed = filesystem.cat_file(location).decode()
    locations = [line.strip() for line in listed.splitlines() if line.strip()]
    if not locations:
        raise FileNotFoundError(f"No FITS locations found in {location}.")
    logger.info(f"Found {len(locations)} FITS locations in {location}.")
    warn_if_compressed(locations)
    return locations


def resolve_fits_list(
    fits: str | Path | list[str],
    storage_options: dict[str, dict] | None = None,
) -> list[str]:
    """Resolve an input specification into a list of FITS locations.

    The same forms serve both access modes. Where the files live and what a
    store does with their pixels are independent choices. A copy store can be
    built from an archive, and a reference store from a disk.

    A specification naming a protocol is resolved through fsspec, so the forms
    hold for any protocol the files sit behind; the list of locations it
    produces may name any mixture of protocols, since each is opened by its own.

    Parameters
    ----------
    fits
        A directory or bucket prefix to search, a single FITS file, a list of
        paths or URLs, or a text file holding one path or URL per line.
    storage_options
        Filesystem options keyed by protocol, as `options_for` takes them. Used
        only where the specification names a server.

    Returns
    -------
    result
        Locations in a stable order.

    Raises
    ------
    TypeError
        Where the specification is none of those things.
    FileNotFoundError
        Where the directory or file does not exist, or no locations were
        found.
    """
    if isinstance(fits, (list, tuple)):
        locations = [str(location) for location in fits]
        if not locations:
            raise FileNotFoundError("No FITS locations given.")
        logger.info(f"Given {len(locations)} FITS locations.")
        return locations

    if not isinstance(fits, (str, Path)):
        raise TypeError(
            f"Expected a directory, a file, or a list of locations, got "
            f"{type(fits).__name__}."
        )

    if is_remote(str(fits)):
        return resolve_remote_fits_list(str(fits), storage_options)

    if os.path.isdir(fits):
        fits_dir = Path(fits)
        fits_files = sorted(
            path
            for path in fits_dir.iterdir()
            if path.is_file()
            and (
                path.suffix.lower() in FITS_SUFFIXES
                or path.suffix.lower() in COMPRESSED_SUFFIXES
            )
        )
        if len(fits_files) == 0:
            raise FileNotFoundError(f"No FITS files found in {fits_dir}.")
        logger.info(f"Found {len(fits_files)} FITS files in {fits_dir}.")
        warn_if_compressed([str(p) for p in fits_files])
        return [str(p) for p in fits_files]

    if not os.path.isfile(fits):
        raise FileNotFoundError(f"{fits} does not exist.")

    if Path(fits).suffix.lower() in FITS_SUFFIXES:
        return [str(fits)]

    with open(fits) as handle:
        locations = [line.strip() for line in handle if line.strip()]
    if len(locations) == 0:
        raise FileNotFoundError(f"No FITS locations found in {fits}.")
    logger.info(f"Found {len(locations)} FITS locations in {Path(fits).name}.")
    warn_if_compressed(locations)
    return locations


def warn_if_compressed(locations: list[str]) -> None:
    """Warn that a compressed input grows on disk once copied.

    Astropy inflates compressed inputs before copy-mode storage. Reference mode
    requires directly addressable pixel ranges.

    Parameters
    ----------
    locations
        Paths or URLs of the FITS files.
    """
    compressed = [location for location in locations if compression_suffix(location)]
    if compressed:
        logger.warning(
            f"{len(compressed)} of {len(locations)} files are compressed. A copy "
            f"store holds their pixels decompressed, so expect it to be several "
            f"times their size on disk. A reference store cannot address them at all."
        )


def resolve_header_workers(
    n_files: int,
    remote: bool,
    max_workers: int | None = None,
) -> int:
    """Choose concurrent FITS header readers.

    Bound remote concurrency by machine capacity, `HEADER_WORKERS`, and the
    file count. Keep local reads serial to avoid disk contention. An explicit
    ``max_workers`` overrides the capacity and shared-service caps.

    Parameters
    ----------
    n_files
        Files to read.
    remote
        Whether any file uses remote storage.
    max_workers
        Override. Clamped to at least one and to the number of files.

    Returns
    -------
    result
        Concurrent header reader count.
    """
    if not remote and max_workers is None:
        return 1
    return polite_workers(
        HEADER_WORKERS,
        requested=max_workers,
        ceiling=n_files,
        minimum=REMOTE_MINIMUM_WORKERS,
    )


def read_header(
    location: str,
    require_referenceable: bool,
    storage_options: dict[str, dict] | None = None,
    attempts: int = HEADER_ATTEMPTS,
) -> tuple[str, dict[str, Any] | None, Exception | None]:
    """Read one file's metadata and return any failure with its location.

    Return worker failures as values so the caller can preserve input order and
    file attribution. Retry remote open failures that may reflect archive load.

    Parameters
    ----------
    location
        Path or URL of the FITS file.
    require_referenceable
        Whether the file must be addressable as byte ranges.
    storage_options
        Filesystem options keyed by protocol, as `options_for` takes them.
    attempts
        How many times to try opening a served file.

    Returns
    -------
    location
        Input location.
    metadata
        None where the file could not be read.
    error
        Why not, where it could not.
    """
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            return (
                str(location),
                extract_metadata(
                    str(location),
                    require_referenceable=require_referenceable,
                    storage_options=storage_options,
                ),
                None,
            )
        except FITSOpenError as error:
            if attempt >= attempts or not is_remote(location):
                return str(location), None, error
            # Exponential jitter prevents simultaneous retry bursts.
            time.sleep(RETRY_PAUSE * 2 ** (attempt - 1) * (0.5 + random.random()))
            logger.debug(f"Retrying {location} ({attempt} of {attempts} failed).")
        except Exception as error:
            return str(location), None, error


def read_header_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Read what an interrupted header pass had recorded.

    A line at a time, so a run killed mid-write loses that record and no other.
    The partial last line will not parse and is dropped. Nothing else in the
    file depends on it, which is the point of a line-delimited checkpoint over
    one large document.

    Parameters
    ----------
    path
        Checkpoint file, one JSON record per line.

    Returns
    -------
    result
        Location to its metadata. Empty where there is no checkpoint.
    """
    if not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    with open(path) as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[record["image_location"]] = record
    return done


def write_header_checkpoint(handle, metadata: dict[str, Any]) -> None:
    """Append one header's metadata to the checkpoint and flush it.

    Flushed every time. A checkpoint that is only worth what was buffered when
    the run died is not worth keeping, and header reads are slow enough that
    the flush adds no measurable time beside them.

    Parameters
    ----------
    handle
        Checkpoint, opened for appending.
    metadata
        One image's metadata, as `read_header` returns it.
    """
    plain = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in metadata.items()
    }
    handle.write(json.dumps(plain) + "\n")
    handle.flush()


def build_header_index(
    fits_locations: list[str],
    require_referenceable: bool = False,
    max_workers: int | None = None,
    checkpoint: Path | None = None,
    storage_options: dict[str, dict] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Extract metadata for every readable image.

    Log unreadable files and retain the remaining catalogue.

    Headers may be read concurrently, but they are recorded in the order given.
    ``Executor.map`` yields results in the order its arguments were submitted,
    so a store's rows follow its input list however the reads interleave.

    Parameters
    ----------
    fits_locations
        Paths or URLs of the FITS files.
    require_referenceable
        Whether every file must be addressable as byte ranges, as a reference
        store needs.
    max_workers
        Headers to read at once. Defaults to `resolve_header_workers`, which
        reads remote files a few at a time and local ones one at a time.
    storage_options
        Filesystem options keyed by protocol, as `options_for` takes them.

    Returns
    -------
    header_index
        Metadata for each usable image, in input order.
    image_valid
        One row per input file recording whether it was usable and why not.
    metadata_valid
        One row per usable image flagging which metadata fields were populated.

    Raises
    ------
    RuntimeError
        Where no file could be read.
    """
    # Remote wait time makes concurrent header reads beneficial in mixed lists.
    remote = any(is_remote(location) for location in fits_locations)

    # Resume from checkpointed headers to avoid repeated remote reads.
    already = read_header_checkpoint(checkpoint) if checkpoint else {}
    outstanding = [location for location in fits_locations if location not in already]
    if already:
        logger.info(
            f"Resuming with {len(already)} header(s) read and "
            f"{len(outstanding)} remaining."
        )
    workers = resolve_header_workers(len(outstanding), remote, max_workers)

    # Position-indexed results preserve input order and support retry gaps.
    metadata_at: dict[int, dict[str, Any]] = {}
    errors_at: dict[int, Exception] = {}

    with ExitStack() as stack:
        # One main-thread warning context avoids races across the worker pool.
        stack.enter_context(quiet_fits_warnings(expect_truncated=True))
        recorder = stack.enter_context(open(checkpoint, "a")) if checkpoint else None
        if workers > 1:
            logger.info(f"Reading {len(outstanding)} headers, {workers} at a time.")
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
            results = pool.map(
                read_header,
                outstanding,
                repeat(require_referenceable),
                repeat(storage_options),
            )
        else:
            results = (
                read_header(location, require_referenceable, storage_options)
                for location in outstanding
            )

        position_of = {
            location: position for position, location in enumerate(fits_locations)
        }
        for location in already:
            if location in position_of:
                metadata_at[position_of[location]] = already[location]

        for location, (_, metadata, error) in zip(
            outstanding,
            tqdm(results, total=len(outstanding), desc="Reading FITS headers"),
            strict=True,
        ):
            position = position_of[location]
            if error is None:
                metadata_at[position] = metadata
                if recorder is not None:
                    write_header_checkpoint(recorder, metadata)
            else:
                # Report failures after closing the progress bar.
                errors_at[position] = error

        # A delayed serial retry tolerates archive failure bursts spanning one pass.
        if errors_at and remote:
            logger.info(f"Reading {len(errors_at)} header(s) again that failed.")
            for position in tqdm(sorted(errors_at), desc="Retrying failed reads"):
                _, metadata, error = read_header(
                    fits_locations[position], require_referenceable, storage_options
                )
                if error is None:
                    metadata_at[position] = metadata
                    del errors_at[position]

    header_index = []
    image_record = []
    metadata_record = []
    n_stokes_assumed = 0

    for position, location in enumerate(fits_locations):
        metadata = metadata_at.get(position)
        if metadata is None:
            image_record.append(
                {
                    "image_location": str(location),
                    "observation_type": None,
                    "converted": False,
                    "error": str(errors_at[position]),
                    "array_index": None,
                }
            )
            continue
        # Count inferred Stokes axes as read diagnostics.
        n_stokes_assumed += bool(metadata.pop("stokes_assumed", False))
        image_record.append(
            {
                "image_location": str(location),
                "observation_type": metadata["observation_type"],
                "converted": True,
                "error": None,
                "array_index": len(header_index),
            }
        )
        metadata_valid = {
            "image_location": metadata["image_location"],
            "observation_type": metadata["observation_type"],
        }
        for k in metadata.keys():
            if k == "image_location":
                continue
            if metadata[k] is None:
                metadata_valid[k] = False
            elif isinstance(metadata[k], str) and metadata[k] == "":
                metadata_valid[k] = False
            elif isinstance(metadata[k], (float, np.floating)) and np.isnan(
                metadata[k]
            ):
                metadata_valid[k] = False
            elif isinstance(metadata[k], (tuple, list)):
                metadata_valid[k] = all(
                    v is not None
                    and not (isinstance(v, (float, np.floating)) and np.isnan(v))
                    for v in metadata[k]
                )
            else:
                metadata_valid[k] = True
        header_index.append(metadata)
        metadata_record.append(metadata_valid)

    n_ok = len(header_index)
    n_all = len(fits_locations)
    if errors_at:
        logger.warning(
            f"Skipped {len(errors_at)} of {n_all} files, which image_valid.csv records."
        )
        by_kind = defaultdict(list)
        for position, error in sorted(errors_at.items()):
            by_kind[type(error).__name__].append((fits_locations[position], error))
        for kind, entries in sorted(by_kind.items(), key=lambda item: -len(item[1])):
            location, error = entries[0]
            logger.warning(
                f"{len(entries)} {kind}. Example {location} failed with {error}."
            )
    if n_stokes_assumed:
        # Aggregate missing Stokes-axis warnings across the survey.
        logger.warning(
            f"No STOKES axis in {n_stokes_assumed} of {n_ok} files, so Stokes I "
            f"is assumed."
        )
    if n_ok == 0:
        raise RuntimeError("No usable FITS files found.")
    image_valid_df = pd.DataFrame(image_record)
    metadata_valid_df = pd.DataFrame(metadata_record)
    return header_index, image_valid_df, metadata_valid_df


def get_max_shape(header_index: list[dict[str, Any]]) -> tuple[int, int]:
    """Return the largest image height and width in a catalogue.

    Parameters
    ----------
    header_index
        Metadata dicts.

    Returns
    -------
    result
        Shape as ``(H, W)``, i.e. (row, column).
    """
    ny = [t["pixel_shape"][0] for t in header_index]
    nx = [t["pixel_shape"][1] for t in header_index]
    return max(ny), max(nx)


def write_image_data_to_zarr_array(
    image_array: zarr.Array,
    header_index: list[dict[str, Any]],
    storage_options: dict[str, dict] | None = None,
) -> None:
    """Read each image's pixels and write them into an array.

    Images smaller than the array are written into its corner; the rest of each
    row keeps the array's fill value, so no explicit padding is needed.

    Parameters
    ----------
    image_array
        Destination, indexed by image along its first axis.
    header_index
        Metadata dicts giving the source location of each image.
    storage_options
        Filesystem options keyed by protocol, as `options_for` takes them.
    """
    for idx, image in enumerate(
        tqdm(header_index, total=len(header_index), desc="Copying pixel data")
    ):
        with quiet_fits_warnings():
            location = image["image_location"]
            # Direct local reads avoid filesystem-abstraction overhead.
            source = (
                open_remote(location, storage_options)
                if is_remote(location)
                else location
            )
            with fits.open(source, memmap=False) as hdul:
                hdu = get_image_hdu(hdul)
                data = extract_stokes_i(hdu.data, hdu.header)
        # The NaN fill value pads differing image dimensions.
        h, w = data.shape[-2:]
        if data.ndim == 2:
            image_array[idx, :h, :w] = data
        else:
            image_array[idx, :, :h, :w] = data


def write_metadata_arrays(
    root: zarr.Group,
    header_index: list[dict[str, Any]],
    metadata_chunk_size: int,
) -> zarr.Group:
    """Write one metadata array per field, with a row per image.

    Chunk ``header`` one element at a time because readers request individual
    headers. Shard those chunks into one metadata file.

    Parameters
    ----------
    root
        Root group of the store.
    header_index
        Metadata dicts.
    metadata_chunk_size
        Rows per chunk; passing the row count keeps each array in one chunk so
        a reader loads it in a single read.

    Returns
    -------
    result
        The metadata group.
    """
    meta = root.require_group("metadata")
    naxes = len(header_index[0]["crval"])
    meta.create_array(
        "crval",
        data=np.array([m["crval"] for m in header_index], dtype=np.float64),
        chunks=(metadata_chunk_size, naxes),
    )
    meta.create_array(
        "crpix",
        data=np.array([m["crpix"] for m in header_index], dtype=np.float64),
        chunks=(metadata_chunk_size, naxes),
    )
    meta.create_array(
        "cd",
        data=np.array([m["cd"] for m in header_index], dtype=np.float64),
        chunks=(metadata_chunk_size, naxes, naxes),
    )
    meta.create_array(
        "ctype",
        data=np.array(
            [m["ctype"] for m in header_index], dtype=np.dtypes.StringDType()
        ),
        chunks=(metadata_chunk_size,),
    )
    meta.create_array(
        "image_location",
        data=np.array(
            [m["image_location"] for m in header_index], dtype=np.dtypes.StringDType()
        ),
        chunks=(metadata_chunk_size,),
    )
    meta.create_array(
        "pixel_shape",
        data=np.array([m["pixel_shape"] for m in header_index], dtype=np.int32),
        chunks=(metadata_chunk_size, 2),
    )
    meta.create_array(
        "pixel_centre",
        data=np.array([m["pixel_centre"] for m in header_index], dtype=np.float64),
        chunks=(metadata_chunk_size, 2),
    )
    meta.create_array(
        "pixel_scale",
        data=np.array([m["pixel_scale"] for m in header_index], dtype=np.float64),
        chunks=(metadata_chunk_size, 2),
    )
    meta.create_array(
        "telescope",
        data=np.array(
            [m["telescope"] for m in header_index], dtype=np.dtypes.StringDType()
        ),
        chunks=(metadata_chunk_size,),
    )
    meta.create_array(
        "date",
        data=np.array([m["date"] for m in header_index], dtype=np.float64),
        chunks=(metadata_chunk_size,),
    )
    meta.create_array(
        "frequency",
        data=np.array([m["frequency"] for m in header_index], dtype=np.float64),
        chunks=(metadata_chunk_size,),
    )
    meta.create_array(
        "bandwidth",
        data=np.array([m["bandwidth"] for m in header_index], dtype=np.float64),
        chunks=(metadata_chunk_size,),
    )
    meta.create_array(
        "bmaj",
        data=np.array([m["bmaj"] for m in header_index], dtype=np.float64),
        chunks=(metadata_chunk_size,),
    )
    meta.create_array(
        "bmin",
        data=np.array([m["bmin"] for m in header_index], dtype=np.float64),
        chunks=(metadata_chunk_size,),
    )
    meta.create_array(
        "bpa",
        data=np.array([m["bpa"] for m in header_index], dtype=np.float64),
        chunks=(metadata_chunk_size,),
    )
    meta.create_array(
        "bunit",
        data=np.array(
            [m["bunit"] for m in header_index], dtype=np.dtypes.StringDType()
        ),
        chunks=(metadata_chunk_size,),
    )
    meta.create_array(
        "equinox",
        data=np.array([m["equinox"] for m in header_index], dtype=np.float64),
        chunks=(metadata_chunk_size,),
    )
    meta.create_array(
        "radesys",
        data=np.array(
            [m["radesys"] for m in header_index], dtype=np.dtypes.StringDType()
        ),
        chunks=(metadata_chunk_size,),
    )
    # One-record chunks keep variable-length provenance reads independent.
    meta.create_array(
        "header",
        data=np.array(
            [m["header"] for m in header_index], dtype=np.dtypes.StringDType()
        ),
        chunks=(1,),
        shards=(max(1, len(header_index)),),
    )
    return meta


# A one-megabyte byte budget balances shard-open overhead with spatial locality.
PIXEL_CHUNK_BYTES: int = 1_000_000
# Enforce a minimum spatial block to bound per-file overhead.
MIN_SPATIAL_CHUNK: int = 64
# Quarter-block inner chunks tighten reads around cutouts spanning a few hundred pixels.
PIXEL_CHUNK_SPLIT: int = 4


def pixel_chunks(
    observation_type: str,
    n_chan: int | None,
    image_shape: tuple[int, int],
    itemsize: int,
    chunk_bytes: int = PIXEL_CHUNK_BYTES,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Choose the chunk and shard shapes of the pixel array.

    Derive shard shape from a byte budget, element type, channel count, and
    image size. Reads open intersecting shards and decompress intersecting
    chunks.

    Every read of a spectral image asks for all channels of a small celestial
    region, so splitting channels across chunks multiplies the number of chunks
    a cutout straddles while reading no fewer pixels. Channels are therefore
    kept together and the celestial block shrinks to stay within the budget.
    Only where that would drive the block below `MIN_SPATIAL_CHUNK` (a cube
    many tens of channels deep) are channels split as well, into equal groups.
    The byte budget determines the channel split continuously.

    Parameters
    ----------
    observation_type
        Survey observation type.
    n_chan
        Channel count for spectral stores.
    image_shape
        Largest image as ``(ny, nx)``. A shard is never much larger than this,
        since there would be nothing to put in the excess.
    itemsize
        Bytes per pixel.
    chunk_bytes
        Target shard size in bytes.

    Returns
    -------
    result
        Chunk shape and shard shape, each matching the pixel array's
        dimensionality.
    """
    ny, nx = image_shape
    channels = 1 if observation_type != "spectral" else n_chan

    side = int(np.sqrt(chunk_bytes / (channels * itemsize)))
    side = max(side, MIN_SPATIAL_CHUNK)
    spatial = (min(side, ny), min(side, nx))
    # Round blocks up to contain a whole number of chunks.
    inner = tuple(int(np.ceil(size / PIXEL_CHUNK_SPLIT)) for size in spatial)
    spatial = tuple(size * PIXEL_CHUNK_SPLIT for size in inner)

    if observation_type != "spectral":
        return (1, *inner), (1, *spatial)

    affordable = int(
        np.clip(chunk_bytes // (spatial[0] * spatial[1] * itemsize), 1, n_chan)
    )
    # Equal channel groups prevent a small remainder from creating another shard.
    groups = int(np.ceil(n_chan / affordable))
    depth = int(np.ceil(n_chan / groups))
    return (1, depth, *inner), (1, depth, *spatial)


def write_reference_map(
    out_path: Path,
    header_index: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the byte map that lets a reference store read its FITS files.

    One record per image, giving where its pixels begin and how they are laid
    out. The ranges a read needs follow from that, so nothing about which bytes
    to fetch is decided here, and a region is turned into ranges when it is
    asked for.

    Parameters
    ----------
    out_path
        Root of the survey store.
    header_index
        Metadata for each image, holding ``data_offset``, ``raw_shape``, and
        ``raw_dtype``.

    Returns
    -------
    result
        What was written, for logging.

    Raises
    ------
    ValueError
        Where any image's pixels cannot be addressed as byte ranges.
    """
    unreferenceable = [
        metadata["image_location"]
        for metadata in header_index
        if metadata.get("data_offset") is None or metadata.get("raw_shape") is None
    ]
    if unreferenceable:
        raise ValueError(
            f"{len(unreferenceable)} file(s) cannot be referenced. The first is "
            f"{unreferenceable[0]}. Scaled pixels or unsupported bit depth may "
            f"be present. Build this survey in copy mode."
        )

    images = [
        {
            "location": str(metadata["image_location"]),
            "offset": int(metadata["data_offset"]),
            "shape": [int(n) for n in metadata["raw_shape"]],
            "dtype": str(metadata["raw_dtype"]),
        }
        for metadata in header_index
    ]
    payload = {"images": images}
    write_json_atomically(Path(out_path) / REFERENCE_MAP_PATH, payload)
    logger.info(f"Byte map contains {len(images)} images.")
    return payload


def extend_reference_map(
    out_path: Path,
    header_index: list[dict[str, Any]],
) -> None:
    """Append images to an existing byte map.

    Parameters
    ----------
    out_path
        Root of the survey store.
    header_index
        Metadata for the images being added.

    Raises
    ------
    FileNotFoundError
        Where the store holds no byte map to extend.
    ValueError
        Where an image's pixels cannot be addressed as byte ranges.
    """
    path = Path(out_path) / REFERENCE_MAP_PATH
    if not path.exists():
        raise FileNotFoundError(f"No byte map at {path} to append to.")
    with open(path) as handle:
        payload = json.load(handle)
    for metadata in header_index:
        if metadata.get("data_offset") is None or metadata.get("raw_shape") is None:
            raise ValueError(
                f"{metadata['image_location']} cannot be referenced. This store "
                f"was built in reference mode and every image must be addressable."
            )
        payload["images"].append(
            {
                "location": str(metadata["image_location"]),
                "offset": int(metadata["data_offset"]),
                "shape": [int(n) for n in metadata["raw_shape"]],
                "dtype": str(metadata["raw_dtype"]),
            }
        )
    write_json_atomically(path, payload)


def create_survey_store(
    header_index: list[dict[str, Any]],
    observation_type: str,
    n_chan: int | None,
    access_mode: str,
    out_path: str | Path,
    chunk_bytes: int,
    image_dtype: np.dtype,
    image_valid_df: pd.DataFrame | None = None,
    metadata_valid_df: pd.DataFrame | None = None,
    storage_options: dict[str, dict] | None = None,
) -> zarr.Group:
    """Write a survey store from an already-validated image catalogue.

    Set ``complete`` after all arrays are written. Write CSV records before the
    pixel pass so interrupted builds retain diagnostics.

    Parameters
    ----------
    header_index
        Metadata dicts for the images to store.
    observation_type
        Type shared by every image.
    n_chan
        Channel count for spectral stores.
    access_mode
        Whether to write pixels into the store or leave them at the source.
    out_path
        Destination, overwritten if it exists.
    chunk_bytes
        Target size in bytes of one shard of the pixel array. Copy mode only:
        a reference store holds no array of its own.
    image_dtype
        Element type of the pixel array.
    image_valid_df, metadata_valid_df
        Records written alongside the store as CSV.
    storage_options
        Filesystem options keyed by protocol, as `options_for` takes them.
        Recorded in the store, less any credential, so that a reference store
        can reconstruct its filesystem configuration.

    Returns
    -------
    result
        Root group of the finished store.
    """
    compressors = [BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")]
    out_path = Path(out_path)
    root = create_store(out_path)
    root.attrs["complete"] = False
    root.attrs["n_images"] = len(header_index)
    root.attrs["observation_type"] = observation_type
    root.attrs["access_mode"] = access_mode
    if n_chan is not None:
        root.attrs["n_chan"] = int(n_chan)
    recorded = persistable_options(storage_options)
    if recorded:
        root.attrs["storage_options"] = recorded
    # Persist provenance before the slower pixel write.
    if image_valid_df is not None:
        image_valid_df.to_csv(out_path / "image_valid.csv", index=False)
    if metadata_valid_df is not None:
        metadata_valid_df.to_csv(out_path / "metadata_valid.csv", index=False)
    if access_mode == "copy":
        logger.info("Creating pixel data array ...")
        ny, nx = get_max_shape(header_index)
        if observation_type == "spectral":
            shape = (len(header_index), n_chan, ny, nx)
        else:
            shape = (len(header_index), ny, nx)
        chunks, shards = pixel_chunks(
            observation_type,
            n_chan,
            (ny, nx),
            np.dtype(image_dtype).itemsize,
            chunk_bytes,
        )
        logger.info(f"Pixel array {shape} in shards of {shards}, chunks of {chunks}.")
        data = root.create_array(
            "data",
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype=image_dtype,
            fill_value=np.nan,
            compressors=compressors,
        )
        write_image_data_to_zarr_array(data, header_index, storage_options)
    else:
        logger.info("Creating byte map ...")
        write_reference_map(out_path, header_index)
    logger.info("Creating metadata arrays ...")
    write_metadata_arrays(root, header_index, metadata_chunk_size=len(header_index))
    root.attrs["complete"] = True
    consolidate(root)
    logger.info(f"Finished building survey store at {out_path}.")
    return root


def build_survey_store(
    fits: str | Path | list[str],
    out_path: str | Path,
    mode: str = "copy",
    chunk_bytes: int = PIXEL_CHUNK_BYTES,
    image_dtype: np.dtype = np.float32,
    max_workers: int | None = None,
    resume: bool = False,
    storage_options: dict[str, dict] | None = None,
) -> zarr.Group:
    """Build a survey store from a set of FITS images.

    Parameters
    ----------
    fits
        A directory or bucket prefix of FITS files, a single FITS file, a list
        of locations, or a text file holding one per line. Either access mode
        accepts any of them, regardless of where the files live.
    out_path
        Destination, overwritten if it exists.
    mode
        Whether to write pixels into the store or leave them at the source and
        fetch regions on demand.
    chunk_bytes
        Target size in bytes of one shard of the pixel array, in copy mode. The
        shard and chunk shapes are derived from it, accounting for the element
        type, the channel count, and the image size. `pixel_chunks` has the detail.
    image_dtype
        Element type of the pixel array.
    max_workers
        Concurrent header readers. Defaults to `resolve_header_workers`.
    storage_options
        Filesystem options keyed by protocol, as ``{'s3': {'anon': True}}``.
        Used for every served file the build touches, and recorded in the store
        for reference reads. Credentials are omitted. See `persistable_options`.

    Returns
    -------
    result
        Root group of the finished store.

    Raises
    ------
    IncompatibleTilesError
        Where the images cannot share one store, in which case nothing is
        written.
    RuntimeError
        Where no file could be read.
    """
    logger.info(f"Building {mode} survey store at {out_path} ...")
    if mode in ("local", "remote"):
        raise ValueError(f"mode must be 'copy' or 'reference', got {mode!r}.")
    if mode not in ("copy", "reference"):
        raise ValueError(f"Expected mode 'copy' or 'reference', got {mode!r}.")
    fits_paths = resolve_fits_list(fits, storage_options)
    # Keep the checkpoint beside the target so store replacement preserves it.
    checkpoint = Path(str(out_path) + HEADER_CHECKPOINT_SUFFIX)
    if not resume:
        checkpoint.unlink(missing_ok=True)
    header_index, image_valid_df, metadata_valid_df = build_header_index(
        fits_paths,
        require_referenceable=(mode == "reference"),
        max_workers=max_workers,
        checkpoint=checkpoint,
        storage_options=storage_options,
    )
    observation_type, n_chan = check_compatibility(header_index)
    root = create_survey_store(
        header_index,
        observation_type,
        n_chan,
        access_mode=mode,
        out_path=out_path,
        chunk_bytes=chunk_bytes,
        image_dtype=image_dtype,
        image_valid_df=image_valid_df,
        metadata_valid_df=metadata_valid_df,
        storage_options=storage_options,
    )
    # Remove the checkpoint only after publishing the completed store.
    checkpoint.unlink(missing_ok=True)
    return root
