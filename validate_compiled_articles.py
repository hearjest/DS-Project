import argparse
import hashlib
import os
import pandas as pd


def hash_val(value: str) -> str:
    """Return a short hash for string values to keep memory light."""
    return hashlib.md5(value.encode("utf-8", errors="ignore")).hexdigest()


def validate_file(csv_path: str, chunksize: int = 50_000):
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    print("=" * 80)
    print("VALIDATING COMPILED ARTICLES")
    print("=" * 80)
    print(f"File: {csv_path}")
    print(f"Chunk size: {chunksize:,}")

    # Overall tracking
    total_rows = 0
    chunk_idx = 0
    expected_cols = None
    column_mismatches = []

    # Null tracking (key fields only)
    key_cols = ["Date", "Article_title", "Url", "Stock_symbol", "Article", "Source_file"]
    null_counts = {col: 0 for col in key_cols}

    # Duplicate tracking (hashes to keep memory lower)
    url_hashes = set()
    title_date_hashes = set()
    url_dupes = 0
    title_date_dupes = 0

    # Date parsing stats
    parsed_dates = 0
    invalid_dates = 0

    # Rows entirely empty (all NaN)
    empty_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        chunk_idx += 1
        total_rows += len(chunk)

        # Column consistency
        if expected_cols is None:
            expected_cols = list(chunk.columns)
        elif list(chunk.columns) != expected_cols:
            column_mismatches.append((chunk_idx, list(chunk.columns)))

        # Empty rows
        empty_rows += chunk.isna().all(axis=1).sum()

        # Null counts for key columns
        for col in key_cols:
            if col in chunk.columns:
                null_counts[col] += chunk[col].isna().sum()

        # Date parsing
        if "Date" in chunk.columns:
            dates = pd.to_datetime(chunk["Date"], errors="coerce", utc=True)
            parsed_dates += dates.notna().sum()
            invalid_dates += dates.isna().sum()

        # Duplicate URL detection
        if "Url" in chunk.columns:
            for val in chunk["Url"].dropna().astype(str):
                h = hash_val(val.strip().lower())
                if h in url_hashes:
                    url_dupes += 1
                else:
                    url_hashes.add(h)

        # Duplicate Title+Date detection
        has_title = "Article_title" in chunk.columns
        has_date = "Date" in chunk.columns
        if has_title and has_date:
            for _, row in chunk[["Article_title", "Date"]].iterrows():
                title = str(row["Article_title"])
                date = str(row["Date"])
                h = hash_val(f"{title.lower()}||{date}")
                if h in title_date_hashes:
                    title_date_dupes += 1
                else:
                    title_date_hashes.add(h)

        if chunk_idx % 20 == 0:
            print(f"[PROGRESS] Chunks: {chunk_idx}, Rows: {total_rows:,}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Rows processed: {total_rows:,}")
    print(f"Columns: {len(expected_cols) if expected_cols else 0}")
    if expected_cols:
        print(f"Column names: {expected_cols}")

    if column_mismatches:
        print(f"\n[WARNING] Column mismatches in {len(column_mismatches)} chunks.")
        for idx, cols in column_mismatches[:5]:
            print(f"  - Chunk {idx} cols ({len(cols)}): {cols}")
    else:
        print("\nColumn consistency: OK")

    print("\nNull counts (key columns):")
    for col, cnt in null_counts.items():
        if col in (expected_cols or []):
            pct = (cnt / total_rows * 100) if total_rows else 0
            print(f"  {col:15s}: {cnt:,} ({pct:.2f}%)")

    print(f"\nEmpty rows (all NaN): {empty_rows:,}")

    print("\nDate parsing:")
    print(f"  Parsed: {parsed_dates:,}")
    print(f"  Invalid: {invalid_dates:,}")

    print("\nDuplicates:")
    print(f"  URL duplicates: {url_dupes:,}")
    print(f"  Title+Date duplicates: {title_date_dupes:,}")

    print("\nSource breakdown (top 10):")
    try:
        source_counts = (
            pd.read_csv(csv_path, usecols=["Source_file"], low_memory=False)
            .dropna()
            .value_counts()
            .head(10)
        )
        print(source_counts)
    except Exception as e:
        print(f"  [WARN] Could not load Source_file counts: {e}")

    print("\n[OK] Validation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate compiled Apple articles CSV.")
    parser.add_argument(
        "--file",
        default="C:/Users/brian/.vscode/dsproject/DS-Project/apple_articles_compiled2.csv",
        help="Path to compiled CSV (default: apple_articles_compiled2.csv)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="Chunk size for reading (default: 50,000)",
    )
    args = parser.parse_args()
    validate_file(args.file, args.chunksize)

