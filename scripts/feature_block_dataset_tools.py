"""Split datasets into feature blocks and combine selected columns again."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".matplotlib_cache").resolve()))

import numpy as np
import pandas as pd

sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_data
from src.config import TARGET_COLUMN as DEFAULT_TARGET_COLUMN
from src.preprocessing import drop_id_column
from src.utils import ensure_dir


# ---------------------------------------------------------------------------
# Easy-to-edit defaults. Command-line arguments can override these values.
# ---------------------------------------------------------------------------
INPUT_FILE = "arcene_data.csv"
NUMBER_OF_BLOCKS = 10
TARGET_COLUMN = DEFAULT_TARGET_COLUMN
ID_COLUMN = "none"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "split_datasets"
SELECTION_PERCENTAGE = 20
SELECTED_FEATURES_GLOB = "**/top_20_max_abs_features.csv"
FINAL_OUTPUT_FILE = "arcene_selected_features_combined.csv"


@dataclass(frozen=True)
class LoadedDataset:
    df: pd.DataFrame
    dataset_base: str
    target_column: str


@dataclass(frozen=True)
class SplitOptions:
    """Inputs required for the feature-block split operation."""

    input_file: str | Path = INPUT_FILE
    number_of_blocks: int = NUMBER_OF_BLOCKS
    target_column: str = TARGET_COLUMN
    output_directory: str | Path = OUTPUT_DIRECTORY
    id_column: str | None = ID_COLUMN
    write_feature_ranking_files: bool = True
    feature_ranking_raw_dir: str | Path = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class CombineOptions:
    """Inputs required for mapping selected block features back to the dataset."""

    input_file: str | Path = INPUT_FILE
    selected_features_dir: str | Path = OUTPUT_DIRECTORY
    selected_features_glob: str = SELECTED_FEATURES_GLOB
    target_column: str = TARGET_COLUMN
    output_file: str | Path = FINAL_OUTPUT_FILE
    id_column: str | None = ID_COLUMN
    mapping_file: str | Path | None = None
    write_feature_ranking_files: bool = True
    feature_ranking_raw_dir: str | Path = PROJECT_ROOT / "data" / "raw"
    selected_feature_paths: list[str | Path] | None = None


def normalize_optional_text(value: str | None) -> str | None:
    """Convert CLI placeholders such as ``none`` into ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "-"}:
        return None
    return text


def dataset_base_name(input_file: str | Path) -> str:
    """Return the dataset name without extension or the ``_data`` suffix."""
    stem = Path(input_file).stem
    if stem.endswith("_data"):
        stem = stem[: -len("_data")]
    return stem


def load_dataset_with_target(
    input_file: str | Path,
    target_column: str,
    id_column: str | None = None,
) -> LoadedDataset:
    """Read a target-containing CSV or the project's raw data/label pair."""
    input_path = Path(input_file)
    dataset_base = dataset_base_name(input_file)

    if input_path.exists():
        df = pd.read_csv(input_path)
        if target_column in df.columns:
            df = drop_id_column(df, id_column=id_column)
            return LoadedDataset(df=df, dataset_base=dataset_base, target_column=target_column)

    try:
        df = load_data(str(input_file), folder="raw", target_column=target_column)
    except Exception as exc:
        raise ValueError(
            f"Dataset yuklenemedi veya target kolonu bulunamadi. "
            f"input_file={input_file}, target_column={target_column}. "
            f"Normal target'li CSV veya proje raw *_data/*_label formati kullanin."
        ) from exc

    df = drop_id_column(df, id_column=id_column)
    return LoadedDataset(df=df, dataset_base=dataset_base, target_column=target_column)


def split_feature_columns_evenly(
    feature_columns: list[str], number_of_blocks: int
) -> list[list[str]]:
    """Divide feature names into equally sized, ordered blocks."""
    if number_of_blocks <= 0:
        raise ValueError("number_of_blocks pozitif olmali.")
    if number_of_blocks > len(feature_columns):
        raise ValueError(
            f"Blok sayisi feature sayisindan buyuk olamaz. "
            f"blocks={number_of_blocks}, features={len(feature_columns)}"
        )
    indices = np.array_split(np.arange(len(feature_columns)), number_of_blocks)
    return [[feature_columns[int(i)] for i in block_indices] for block_indices in indices]


def write_feature_ranking_pair(
    block_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    dataset_base: str,
    block_number: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the headerless data/label pair expected by FeatureRank."""
    data_path = output_dir / f"{dataset_base}_block_{block_number:02d}_data.csv"
    label_path = output_dir / f"{dataset_base}_block_{block_number:02d}_label.csv"
    block_df[feature_columns].to_csv(data_path, index=False, header=False)
    block_df[[target_column]].to_csv(label_path, index=False, header=False)
    return data_path, label_path


def write_named_feature_ranking_pair(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    output_stem: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write a headerless pair using an explicit output stem."""
    stem = output_stem
    if stem.endswith("_data"):
        stem = stem[: -len("_data")]
    data_path = output_dir / f"{stem}_data.csv"
    label_path = output_dir / f"{stem}_label.csv"
    df[feature_columns].to_csv(data_path, index=False, header=False)
    df[[target_column]].to_csv(label_path, index=False, header=False)
    return data_path, label_path


# Backward-compatible names used by older notebooks and shell helpers.
write_run_autoencoder_pair = write_feature_ranking_pair
write_named_run_autoencoder_pair = write_named_feature_ranking_pair


def split_dataset_by_features(options: SplitOptions) -> dict:
    """Create block CSVs and the mapping from local to original features."""
    loaded = load_dataset_with_target(
        input_file=options.input_file,
        target_column=options.target_column,
        id_column=normalize_optional_text(options.id_column),
    )
    df = loaded.df
    output_dir = Path(options.output_directory)
    ensure_dir(output_dir)

    if loaded.target_column not in df.columns:
        raise ValueError(f"Target kolonu bulunamadi: {loaded.target_column}")

    feature_columns = [col for col in df.columns if col != loaded.target_column]
    blocks = split_feature_columns_evenly(feature_columns, options.number_of_blocks)
    summary_rows: list[dict] = []
    mapping_rows: list[dict] = []

    raw_dir = Path(options.feature_ranking_raw_dir)
    if options.write_feature_ranking_files:
        ensure_dir(raw_dir)

    original_feature_lookup = {
        feature_name: idx + 1 for idx, feature_name in enumerate(feature_columns)
    }
    sample_count = len(df)

    for block_number, block_features in enumerate(blocks, start=1):
        block_file = output_dir / f"{loaded.dataset_base}_block_{block_number:02d}.csv"
        block_df = df[block_features + [loaded.target_column]].copy()
        block_df.to_csv(block_file, index=False)

        pair_data_path = None
        pair_label_path = None
        if options.write_feature_ranking_files:
            pair_data_path, pair_label_path = write_feature_ranking_pair(
                block_df=block_df,
                feature_columns=block_features,
                target_column=loaded.target_column,
                dataset_base=loaded.dataset_base,
                block_number=block_number,
                output_dir=raw_dir,
            )

        first_feature = block_features[0]
        last_feature = block_features[-1]
        first_index = original_feature_lookup[first_feature]
        last_index = original_feature_lookup[last_feature]
        summary_rows.append(
            {
                "Block": block_number,
                "Sample Count": sample_count,
                "Feature Count": len(block_features),
                "Feature Range": f"{first_index}-{last_index}",
                "First Feature": first_feature,
                "Last Feature": last_feature,
                "File": str(block_file),
                "RunAutoencoderDataFile": str(pair_data_path) if pair_data_path else "",
                "RunAutoencoderLabelFile": (str(pair_label_path) if pair_label_path else ""),
            }
        )

        for local_index, feature_name in enumerate(block_features, start=1):
            mapping_rows.append(
                {
                    "Original_Index": original_feature_lookup[feature_name],
                    "Feature_Name": feature_name,
                    "Block": block_number,
                    "Block_Local_Index": local_index,
                }
            )

        print(f"\nBlock {block_number:02d}")
        print(f"Samples: {sample_count}")
        print(f"Features: {len(block_features)}")
        print(f"Feature range: {first_index}-{last_index}")
        print(f"Saved: {block_file}")
        if options.write_feature_ranking_files:
            print(f"FeatureRank data : {pair_data_path}")
            print(f"FeatureRank label: {pair_label_path}")

    summary_df = pd.DataFrame(summary_rows)
    mapping_df = pd.DataFrame(mapping_rows)
    summary_path = output_dir / "split_summary.csv"
    mapping_path = output_dir / "feature_block_mapping.csv"
    summary_df.to_csv(summary_path, index=False)
    mapping_df.to_csv(mapping_path, index=False)

    total_features_across_blocks = int(summary_df["Feature Count"].sum())
    missing_features = sorted(set(feature_columns) - set(mapping_df["Feature_Name"]))
    duplicate_features = mapping_df["Feature_Name"][
        mapping_df["Feature_Name"].duplicated()
    ].tolist()
    sample_count_preserved = bool((summary_df["Sample Count"] == sample_count).all())

    print("\nSplit verification:")
    print(f"Original samples : {sample_count}")
    print(f"Original features: {len(feature_columns)}")
    print(f"Number of blocks : {options.number_of_blocks}")
    print(f"Total features across blocks: {total_features_across_blocks}")
    print(f"Missing features: {len(missing_features)}")
    print(f"Duplicate features: {len(duplicate_features)}")
    print(f"Sample count preserved: {'YES' if sample_count_preserved else 'NO'}")
    print(f"Summary saved: {summary_path}")
    print(f"Mapping saved: {mapping_path}")

    return {
        "summary_path": summary_path,
        "mapping_path": mapping_path,
        "block_files": [Path(row["File"]) for row in summary_rows],
        "run_autoencoder_data_files": [
            Path(row["RunAutoencoderDataFile"])
            for row in summary_rows
            if row["RunAutoencoderDataFile"]
        ],
    }


def read_selected_feature_file(path: Path) -> list[str]:
    """Read feature names from one ranking CSV."""
    df = pd.read_csv(path)
    candidates = [
        "feature_name",
        "Feature_Name",
        "selected_feature",
        "Selected_Feature",
        "feature",
        "Feature",
    ]
    column = next((name for name in candidates if name in df.columns), None)
    if column is None:
        raise ValueError(
            f"Secilmis feature dosyasinda feature kolonu bulunamadi: {path}. "
            f"Beklenen kolonlardan biri: {candidates}"
        )
    return [str(value) for value in df[column].dropna().tolist()]


def expand_selected_feature_glob(selected_features_glob: str) -> list[str]:
    patterns = [selected_features_glob]

    if "/metrics/" in selected_features_glob:
        patterns.append(selected_features_glob.replace("/metrics/", "/**/"))

    filename = Path(selected_features_glob).name
    parent_pattern = str(Path(selected_features_glob).parent)
    if filename.startswith("top_") and "_max_abs_features.csv" in filename:
        if parent_pattern not in {"", "."}:
            patterns.append(f"{parent_pattern}/**/{filename}")
        patterns.append(f"**/{filename}")

    unique_patterns: list[str] = []
    for pattern in patterns:
        if pattern not in unique_patterns:
            unique_patterns.append(pattern)
    return unique_patterns


def infer_block_source_from_path(path: Path) -> tuple[str, int | None]:
    matches: list[tuple[str, int]] = []
    for part in path.parts:
        match = re.search(r"(.+_block_(\d+)_data)", part)
        if match:
            matches.append((match.group(1), int(match.group(2))))
    if matches:
        return min(matches, key=lambda item: len(item[0]))
    return path.parent.name, None


def local_feature_index(feature_name: str) -> int | None:
    match = re.search(r"(?:^|_)feature_(\d+)$", str(feature_name))
    if match:
        return int(match.group(1))
    match = re.search(r"^F(\d+)$", str(feature_name))
    if match:
        return int(match.group(1))
    return None


def collect_selected_features(
    selected_features_dir: str | Path,
    selected_features_glob: str,
    selected_feature_paths: list[str | Path] | None = None,
) -> pd.DataFrame:
    """Find ranking files and keep their source/order metadata."""
    root = Path(selected_features_dir)
    if selected_feature_paths is not None:
        paths = [Path(path) for path in selected_feature_paths]
    else:
        patterns = expand_selected_feature_glob(selected_features_glob)
        paths = []
        for pattern in patterns:
            paths.extend(sorted(root.glob(pattern)))
            if paths:
                break
    if not paths:
        patterns = expand_selected_feature_glob(selected_features_glob)
        tried = "\n".join(f"  - {root}/{pattern}" for pattern in patterns)
        raise FileNotFoundError(
            f"Secilmis feature dosyasi bulunamadi. Denenen patternler:\n{tried}"
        )

    rows: list[dict] = []
    seen: set[str] = set()
    duplicate_count = 0
    for path in paths:
        features = read_selected_feature_file(path)
        block_name, block_number = infer_block_source_from_path(path)
        class_source = path.parent.name if path.parent.name != block_name else ""
        for order, feature_name in enumerate(features, start=1):
            if feature_name in seen:
                duplicate_count += 1
            seen.add(feature_name)
            rows.append(
                {
                    "Selected_File": str(path),
                    "Block_Source": block_name,
                    "Block_Number": block_number,
                    "Class_Source": class_source,
                    "Order_In_File": order,
                    "Feature_Name": feature_name,
                    "Local_Feature_Index": local_feature_index(feature_name),
                }
            )

    selected_df = pd.DataFrame(rows)
    selected_df.attrs["duplicate_count"] = duplicate_count
    return selected_df


def translate_block_local_features_to_original(
    selected_df: pd.DataFrame,
    mapping_file: str | Path | None,
) -> pd.DataFrame:
    """Use the block mapping to translate local feature names to originals."""
    if not mapping_file:
        return selected_df

    mapping_path = Path(mapping_file)
    if not mapping_path.exists():
        return selected_df

    mapping_df = pd.read_csv(mapping_path)
    required_columns = {"Feature_Name", "Block", "Block_Local_Index"}
    if not required_columns.issubset(mapping_df.columns):
        return selected_df

    translated = selected_df.copy()
    translated["Local_Feature_Name"] = translated["Feature_Name"]

    lookup = {
        (int(row["Block"]), int(row["Block_Local_Index"])): str(row["Feature_Name"])
        for _, row in mapping_df.iterrows()
    }

    original_names: list[str] = []
    translated_count = 0
    for _, row in translated.iterrows():
        block_number = row.get("Block_Number")
        local_index = row.get("Local_Feature_Index")
        if pd.notna(block_number) and pd.notna(local_index):
            original_name = lookup.get((int(block_number), int(local_index)))
            if original_name:
                original_names.append(original_name)
                translated_count += 1
                continue
        original_names.append(str(row["Feature_Name"]))

    translated["Feature_Name"] = original_names
    translated.attrs["duplicate_count"] = selected_df.attrs.get("duplicate_count", 0)
    translated.attrs["translated_count"] = translated_count
    return translated


def combine_selected_features(options: CombineOptions) -> dict:
    """Build one dataset from selected features and preserve mapping details."""
    loaded = load_dataset_with_target(
        input_file=options.input_file,
        target_column=options.target_column,
        id_column=normalize_optional_text(options.id_column),
    )
    df = loaded.df
    feature_columns = [col for col in df.columns if col != loaded.target_column]

    selected_df = collect_selected_features(
        selected_features_dir=options.selected_features_dir,
        selected_features_glob=options.selected_features_glob,
        selected_feature_paths=options.selected_feature_paths,
    )
    selected_df = translate_block_local_features_to_original(
        selected_df=selected_df,
        mapping_file=options.mapping_file,
    )

    selected_features_raw = selected_df["Feature_Name"].tolist()
    selected_features = list(dict.fromkeys(selected_features_raw))
    duplicate_selected_count = len(selected_features_raw) - len(selected_features)
    missing_features = [feature for feature in selected_features if feature not in feature_columns]
    if missing_features:
        preview = missing_features[:20]
        raise ValueError(
            f"Secilen feature'lar orijinal dataset icinde bulunamadi. "
            f"Missing count={len(missing_features)}, preview={preview}"
        )

    output_path = Path(options.output_file)
    ensure_dir(output_path.parent)
    final_df = df[selected_features + [loaded.target_column]].copy()
    final_df.to_csv(output_path, index=False)

    pair_data_path = None
    pair_label_path = None
    if options.write_feature_ranking_files:
        raw_dir = Path(options.feature_ranking_raw_dir)
        ensure_dir(raw_dir)
        pair_data_path, pair_label_path = write_named_feature_ranking_pair(
            df=final_df,
            feature_columns=selected_features,
            target_column=loaded.target_column,
            output_stem=output_path.stem,
            output_dir=raw_dir,
        )

    selected_detail_path = output_path.parent / f"{output_path.stem}_selected_feature_details.csv"
    selected_df.to_csv(selected_detail_path, index=False)

    if options.mapping_file:
        mapping_path = Path(options.mapping_file)
        if mapping_path.exists():
            mapping_df = pd.read_csv(mapping_path)
            selected_mapping = mapping_df[mapping_df["Feature_Name"].isin(selected_features)].copy()
            selected_mapping.to_csv(
                output_path.parent / f"{output_path.stem}_selected_feature_mapping.csv",
                index=False,
            )

    block_counts = (
        selected_df.groupby("Block_Source")["Feature_Name"]
        .nunique()
        .reset_index(name="Selected_Feature_Count")
        .sort_values("Block_Source")
    )
    block_counts_path = output_path.parent / f"{output_path.stem}_selected_counts_by_block.csv"
    block_counts.to_csv(block_counts_path, index=False)

    print("\nCombine verification:")
    print(f"Original samples : {len(df)}")
    print(f"Original features: {len(feature_columns)}")
    print(f"Selected feature files: {selected_df['Selected_File'].nunique()}")
    print(f"Translated local block features: {selected_df.attrs.get('translated_count', 0)}")
    print(f"Total selected features: {len(selected_features)}")
    print(f"Duplicate selected features: {duplicate_selected_count}")
    print(f"Missing selected features: {len(missing_features)}")
    print(f"Final samples : {len(final_df)}")
    print(f"Final features: {len(selected_features)}")
    print("Target included: YES")
    print(f"Saved: {output_path}")
    if options.write_feature_ranking_files:
        print(f"FeatureRank data : {pair_data_path}")
        print(f"FeatureRank label: {pair_label_path}")
    print(f"Details saved: {selected_detail_path}")
    print(f"Block counts saved: {block_counts_path}")

    return {
        "output_path": output_path,
        "selected_detail_path": selected_detail_path,
        "block_counts_path": block_counts_path,
        "selected_feature_count": len(selected_features),
        "duplicate_selected_feature_count": duplicate_selected_count,
        "run_autoencoder_data_path": pair_data_path,
        "run_autoencoder_label_path": pair_label_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature kolonlarini bloklara boler ve secilmis feature'lari tekrar birlestirir."
    )
    parser.add_argument(
        "--mode", choices=["split", "combine", "split-and-combine"], default="split"
    )
    parser.add_argument("--input-file", default=INPUT_FILE)
    parser.add_argument("--number-of-blocks", type=int, default=NUMBER_OF_BLOCKS)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--id-column", default=ID_COLUMN)
    parser.add_argument("--output-directory", default=OUTPUT_DIRECTORY)
    parser.add_argument("--selection-percentage", type=float, default=SELECTION_PERCENTAGE)
    parser.add_argument("--selected-features-dir", default=OUTPUT_DIRECTORY)
    parser.add_argument("--selected-features-glob", default=None)
    parser.add_argument("--final-output-file", default=FINAL_OUTPUT_FILE)
    parser.add_argument("--mapping-file", default=None)
    parser.add_argument(
        "--write-feature-ranking-files",
        "--write-run-autoencoder-files",
        dest="write_feature_ranking_files",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--feature-ranking-raw-dir",
        "--run-autoencoder-raw-dir",
        dest="feature_ranking_raw_dir",
        default=str(PROJECT_ROOT / "data" / "raw"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_glob = args.selected_features_glob
    if selected_glob is None:
        percent_tag = (
            int(args.selection_percentage)
            if float(args.selection_percentage).is_integer()
            else args.selection_percentage
        )
        selected_glob = f"**/top_{percent_tag}_max_abs_features.csv"

    split_result = None
    if args.mode in {"split", "split-and-combine"}:
        split_result = split_dataset_by_features(
            SplitOptions(
                input_file=args.input_file,
                number_of_blocks=args.number_of_blocks,
                target_column=args.target_column,
                output_directory=args.output_directory,
                id_column=args.id_column,
                write_feature_ranking_files=args.write_feature_ranking_files,
                feature_ranking_raw_dir=args.feature_ranking_raw_dir,
            )
        )

    if args.mode in {"combine", "split-and-combine"}:
        mapping_file = args.mapping_file
        if mapping_file is None and split_result is not None:
            mapping_file = str(split_result["mapping_path"])
        combine_selected_features(
            CombineOptions(
                input_file=args.input_file,
                selected_features_dir=args.selected_features_dir,
                selected_features_glob=selected_glob,
                target_column=args.target_column,
                output_file=args.final_output_file,
                id_column=args.id_column,
                mapping_file=mapping_file,
                write_feature_ranking_files=args.write_feature_ranking_files,
                feature_ranking_raw_dir=args.feature_ranking_raw_dir,
            )
        )


if __name__ == "__main__":
    main()
