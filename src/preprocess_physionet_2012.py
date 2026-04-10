from __future__ import annotations

from tqdm import tqdm
import os
import pandas as pd
import pickle


RAW_DATA_PATH = '../../physionet2012'
filepath = '../../data/processed/physionet2012_ts_oc_ids.pkl'
makedir_process = '../../data/processed'


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


def preprocessing(path: str):
    print("=== Starting PhysioNet 2012 preprocessing ===")
    print(f"[1/4] Reading time-series files from: {os.path.abspath(path)}")
    ts = pd.concat([read_ts(path, set_name)
                    for set_name in ['a','b','c']])
    print(f"      Combined time-series rows: {len(ts):,}")

    print("[2/4] Reading outcome tables")
    oc = pd.concat([read_outcomes(path, set_name)
                    for set_name in ['a','b','c']])
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


# Process and Store Data.
ts, oc, ts_ids = preprocessing(RAW_DATA_PATH)
print(f"[4/4] Saving processed artifact to: {os.path.abspath(filepath)}")
os.makedirs(makedir_process, exist_ok=True)
with open(filepath, 'wb') as file:
    pickle.dump([ts, oc, ts_ids], file)
print(
    f"Saved processed PhysioNet artifact: ts rows={len(ts):,}, oc rows={len(oc):,}, "
    f"patients={len(ts_ids):,}"
)
