"""Convert the Kaggle Rio bus GPS CSV into the Parquet file used by rio_segments.py.

Source: https://www.kaggle.com/datasets/igorbalteiro/gps-data-from-rio-de-janeiro-buses
(file ``bus_gps_data/treatedBusDataOnlyRoute.csv``, about 3.8 GB).

Usage:
    uv run python src/convert_gps.py path/to/treatedBusDataOnlyRoute.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import structlog

log = structlog.get_logger()

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "dados"
    / "6-kaggle-rio"
    / "gps_rio_2019.parquet"
)

QUERY = """
COPY (
    SELECT
        strptime(date || ' ' || time, '%m-%d-%Y %H:%M:%S') AS ts,
        "order" AS veic,
        regexp_replace(line, '\\.0$', '') AS linha,
        latitude AS lat,
        longitude AS lon,
        speed
    FROM read_csv(
        ?, delim = ',', quote = '"', header = true,
        ignore_errors = true, strict_mode = false,
        columns = {
            'date': 'VARCHAR', 'time': 'VARCHAR', 'order': 'VARCHAR',
            'line': 'VARCHAR', 'latitude': 'DOUBLE', 'longitude': 'DOUBLE',
            'speed': 'DOUBLE'
        }
    )
) TO '{output}' (FORMAT parquet)
"""


def convert(csv_path: Path, output_path: Path) -> None:
    """Write the cleaned Parquet file.

    Args:
        csv_path: Kaggle CSV with columns date, time, order, line, latitude,
            longitude, speed.
        output_path: Destination Parquet file.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        duckdb.IOException: If DuckDB cannot read the CSV or write the Parquet.
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb.execute(QUERY.replace("{output}", str(output_path)), [str(csv_path)])
    log.info("gps_converted", source=str(csv_path), output=str(output_path))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Convert Kaggle GPS CSV to Parquet.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    convert(args.csv, args.output)


if __name__ == "__main__":
    main()
