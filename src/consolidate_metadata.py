# src/consolidate_metadata.py
#
# Prepares the metadata CSV for the Spark OCR pipeline.
#
# The source file (memes_metadata.csv) is already consolidated — it contains
# all subreddits in a single file with a `Subreddit` column.
# This script normalises column names to lowercase so they match the join key
# used in ocr_pipeline.py (`filename`).
#
# Usage:
#   docker compose exec jupyter python /workspace/src/consolidate_metadata.py

import pandas as pd
import os

METADATA_DIR = "/workspace/data/metadata"
SOURCE_FILE = "memes_metadata.csv"
OUTPUT_PATH = "/workspace/data/metadata_consolidated.csv"


def main():
    source_path = os.path.join(METADATA_DIR, SOURCE_FILE)

    if not os.path.exists(source_path):
        print(f"Source file not found: {source_path}")
        return

    df = pd.read_csv(source_path)
    print(f"Loaded {len(df)} rows from {SOURCE_FILE}")
    print(f"Original columns: {df.columns.tolist()}")

    # Normalise all column names to lowercase so the Spark join on 'filename' works.
    # The source CSV has 'Filename' (capital F) which would cause zero join matches.
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
    print(f"Normalised columns: {df.columns.tolist()}")

    # Sanity checks
    if "filename" not in df.columns:
        print("ERROR: no 'filename' column found after normalisation. Check source CSV.")
        return
    if "subreddit" not in df.columns:
        print("WARNING: no 'subreddit' column found — downstream grouping may fail.")

    null_fn = df["filename"].isna().sum()
    if null_fn:
        print(f"WARNING: {null_fn} rows have null filenames — they will not join.")

    df.to_csv(OUTPUT_PATH, index=False)
    print(
        f"\nOutput: {OUTPUT_PATH}"
        f"\nRows: {len(df)}"
        f"\nColumns: {df.columns.tolist()}"
    )


if __name__ == "__main__":
    main()
