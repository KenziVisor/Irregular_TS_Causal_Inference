from __future__ import annotations

import argparse
import sys
from pathlib import Path

if "--validate-config-only" in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset_config import maybe_run_validate_config_only

    maybe_run_validate_config_only(
        "src/preprocess_physionet_2012.py",
        fixed_dataset="physionet",
    )

from tqdm import tqdm
import os
import pandas as pd
import pickle

from dataset_config import get_config_list, get_config_scalar, get_first_available, load_dataset_config


RAW_DATA_PATH = '../../physionet2012'
filepath = '../../data/processed/physionet2012_ts_oc_ids.pkl'
makedir_process = '../../data/processed'
PHYSIONET_SET_NAMES = ['a', 'b', 'c']


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw PhysioNet 2012 files.")
    parser.add_argument("--dataset-config-csv", default=None)
    parser.add_argument("--raw-data-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Resolve dataset config values and exit without loading data.",
    )
    return parser.parse_args()


def read_ts(raw_data_path, set_name):
    set_dir = raw_data_path + '/set-' + set_name
    files = sorted(os.listdir(set_dir))
    print(
        f"      [set-{set_name}] Reading {len(files):,} patient files from: "
        f"{os.path.abspath(set_dir)}"
    )
    ts = []
    kept_files = 0
    skipped_short = 0
    pbar = tqdm(files, desc='Reading time series set '+set_name, unit='file')
    for f in pbar:
        data = pd.read_csv(set_dir + '/' + f).iloc[1:]
        data = data.loc[data.Parameter.notna()]
        if len(data) <= 5:
            skipped_short += 1
            continue
        data = data.loc[data.Value >= 0]  # neg Value indicates missingness.
        data['RecordID'] = f[:-4]
        ts.append(data)
        kept_files += 1
    ts = pd.concat(ts)
    ts.Time = ts.Time.apply(lambda x:int(x[:2])*60
                            +int(x[3:])) # No. of minutes since admission.
    ts.rename(columns={'Time':'minute', 'Parameter':'variable',
                       'Value':'value', 'RecordID':'ts_id'}, inplace=True)
    print(
        f"      [set-{set_name}] Finished: kept {kept_files:,} / {len(files):,} files | "
        f"rows after cleaning={len(ts):,} | skipped_short={skipped_short:,}"
    )
    return ts


def read_outcomes(raw_data_path, set_name):
    outcome_path = raw_data_path + '/Outcomes-' + set_name + '.txt'
    print(f"      [set-{set_name}] Loading outcomes from: {os.path.abspath(outcome_path)}")
    oc = pd.read_csv(raw_data_path+'/Outcomes-'+set_name+'.txt',
                     usecols=['RecordID', 'Length_of_stay', 'In-hospital_death'])
    oc['subset'] = set_name
    oc.RecordID = oc.RecordID.astype(str)
    oc.rename(columns={'RecordID':'ts_id', 'Length_of_stay':'length_of_stay',
                       'In-hospital_death':'in_hospital_mortality'}, inplace=True)
    print(f"      [set-{set_name}] Outcomes rows loaded: {len(oc):,}")
    return oc


def preprocessing(path: str, set_names=None):
    if set_names is None:
        set_names = PHYSIONET_SET_NAMES
    print("=== Starting PhysioNet 2012 preprocessing ===")
    print(f"[1/4] Reading time-series files from: {os.path.abspath(path)}")
    ts = pd.concat([read_ts(path, set_name)
                    for set_name in set_names])
    print(f"      Combined time-series rows: {len(ts):,}")

    print("[2/4] Reading outcome tables")
    oc = pd.concat([read_outcomes(path, set_name)
                    for set_name in set_names])
    print(f"      Combined outcome rows: {len(oc):,}")

    print("[3/4] Harmonizing patient ids, dropping duplicates, and encoding ICUType")
    ts_ids = sorted(list(ts.ts_id.unique()))
    oc = oc.loc[oc.ts_id.isin(ts_ids)]

    # Drop duplicates.
    ts = ts.drop_duplicates()

    # Convert categorical to numeric.
    ii = (ts.variable=='ICUType')
    for val in [4,3,2,1]:
        kk = ii&(ts.value==val)
        ts.loc[kk, 'variable'] = 'ICUType_'+str(val)
    ts.loc[ii, 'value'] = 1

    print(
        f"      Preprocessing complete: ts shape={ts.shape}, oc shape={oc.shape}, "
        f"patients={len(ts_ids):,}"
    )

    return ts, oc, ts_ids


def main():
    args = parse_args()
    config = load_dataset_config("physionet", args.dataset_config_csv)

    raw_data_path = args.raw_data_path or str(
        get_first_available(
            config,
            ["PREPROCESS_RAW_DATA_PATH", "RAW_DATA_PATH"],
            RAW_DATA_PATH,
        )
    )
    processed_dir = args.processed_dir or str(
        get_config_scalar(config, "PREPROCESS_OUTPUT_DIR", makedir_process)
    )
    output_path = args.output_path or str(
        get_config_scalar(config, "PREPROCESS_OUTPUT_PATH", filepath)
    )
    if args.processed_dir is not None and args.output_path is None:
        output_path = os.path.join(processed_dir, os.path.basename(output_path))
    set_names = list(
        get_config_list(config, "PHYSIONET_SET_NAMES", PHYSIONET_SET_NAMES) or []
    )

    ts, oc, ts_ids = preprocessing(raw_data_path, set_names=set_names)
    print(f"[4/4] Saving processed artifact to: {os.path.abspath(output_path)}")
    os.makedirs(processed_dir, exist_ok=True)
    with open(output_path, 'wb') as file:
        pickle.dump([ts, oc, ts_ids], file)
    print(
        f"Saved processed PhysioNet artifact: ts rows={len(ts):,}, oc rows={len(oc):,}, "
        f"patients={len(ts_ids):,}"
    )


if __name__ == "__main__":
    main()
