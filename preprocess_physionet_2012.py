from __future__ import annotations

from tqdm import tqdm
import os
import pandas as pd
import pickle
import numpy as np
from typing import Iterable, List, Optional, Tuple, Literal, Callable
import operator


RAW_DATA_PATH = '../physionet2012'
CheckMode = Literal["avg_value", "time_spacing", "both"]


def read_ts(raw_data_path, set_name):
    ts = []
    pbar = tqdm(os.listdir(raw_data_path+'/set-'+set_name),
                desc='Reading time series set '+set_name)
    for f in pbar:
        data = pd.read_csv(raw_data_path+'/set-'+set_name+'/'+f).iloc[1:]
        data = data.loc[data.Parameter.notna()]
        if len(data)<=5:
            continue
        data = data.loc[data.Value>=0] # neg Value indicates missingness.
        data['RecordID'] = f[:-4]
        ts.append(data)
    ts = pd.concat(ts)
    ts.Time = ts.Time.apply(lambda x:int(x[:2])*60
                            +int(x[3:])) # No. of minutes since admission.
    ts.rename(columns={'Time':'minute', 'Parameter':'variable',
                       'Value':'value', 'RecordID':'ts_id'}, inplace=True)
    return ts


def read_outcomes(raw_data_path, set_name):
    oc = pd.read_csv(raw_data_path+'/Outcomes-'+set_name+'.txt',
                     usecols=['RecordID', 'Length_of_stay', 'In-hospital_death'])
    oc['subset'] = set_name
    oc.RecordID = oc.RecordID.astype(str)
    oc.rename(columns={'RecordID':'ts_id', 'Length_of_stay':'length_of_stay',
                       'In-hospital_death':'in_hospital_mortality'}, inplace=True)
    return oc


def split_by_T(
    ts: pd.DataFrame,
    ts_ids: Iterable[str],
    *,
    mode: CheckMode,
    var_name: str,
    # thresholds (only used if relevant to mode)
    avg_value_threshold: Optional[float] = None,
    spacing_threshold: Optional[float] = None,
    # optional time window (minutes)
    start_minute: Optional[int] = None,
    end_minute: Optional[int] = None,
    # comparators (can override)
    value_cmp: Callable[[float, float], bool] = operator.ge,   # mean >= threshold
    spacing_cmp: Callable[[float, float], bool] = operator.le, # mean Δt <= threshold
) -> Tuple[List[str], List[str]]:
    """
    Split ids into t1 / t0 based on criteria T computed per ts_id for a single variable.

    ts: DataFrame with columns: ['minute', 'variable', 'value', 'ts_id']
    ts_ids: iterable of ids to evaluate
    mode:
      - "avg_value": use mean(value) for var_name in window
      - "time_spacing": use mean spacing between consecutive measurement times
      - "both": must satisfy BOTH criteria (AND)
    start_minute/end_minute:
      - None, None => full time
      - start only => minute >= start
      - end only   => minute <= end
      - both       => start <= minute <= end

    If there are no samples for (ts_id, var_name) in the window:
      -> goes to t0.
    If time_spacing is requested but there are <2 timestamps:
      -> goes to t0.
    """

    required_cols = {"minute", "variable", "value", "ts_id"}
    missing = required_cols - set(ts.columns)
    if missing:
        raise ValueError(f"ts is missing required columns: {missing}")

    if mode in ("avg_value", "both") and avg_value_threshold is None:
        raise ValueError("avg_value_threshold must be provided for mode 'avg_value' or 'both'.")

    if mode in ("time_spacing", "both") and spacing_threshold is None:
        raise ValueError("spacing_threshold must be provided for mode 'time_spacing' or 'both'.")

    t1: List[str] = []
    t0: List[str] = []

    # Pre-filter to the variable once (huge speedup)
    df = ts.loc[ts["variable"] == var_name, ["ts_id", "minute", "value"]].copy()

    # Apply time window filter
    if start_minute is not None:
        df = df.loc[df["minute"] >= start_minute]
    if end_minute is not None:
        df = df.loc[df["minute"] <= end_minute]

    # Group once
    grouped = df.groupby("ts_id", sort=False)

    for _id in ts_ids:
        _id_str = str(_id)

        if _id_str not in grouped.groups:
            t0.append(_id_str)
            continue

        g = grouped.get_group(_id_str)

        ok_value = True
        ok_spacing = True

        if mode in ("avg_value", "both"):
            mean_val = float(g["value"].mean())
            ok_value = value_cmp(mean_val, float(avg_value_threshold))

        if mode in ("time_spacing", "both"):
            times = np.sort(g["minute"].to_numpy(dtype=float))
            if times.size < 2:
                ok_spacing = False
            else:
                mean_dt = float(np.diff(times).mean())
                ok_spacing = spacing_cmp(mean_dt, float(spacing_threshold))

        ok = (ok_value and ok_spacing) if mode == "both" else (ok_value if mode == "avg_value" else ok_spacing)

        (t1 if ok else t0).append(_id_str)

    return t1, t0


def load_dataset(path: str):
    ts = pd.concat([read_ts(path, set_name)
                    for set_name in ['a','b','c']])
    oc = pd.concat([read_outcomes(path, set_name)
                    for set_name in ['a','b','c']])
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

    return ts, oc, ts_ids


filepath = '../data/processed/physionet2012_ts_oc_ids.pkl'
'''
# Process and Store Data.
ts, oc, ts_ids = load_dataset(RAW_DATA_PATH)
os.makedirs('../data/processed', exist_ok=True)
pickle.dump([ts, oc, ts_ids],
            open(filepath, 'wb'))
'''
# Load Data.
ts, oc, ts_ids = pickle.load(open(filepath, 'rb'))

# Generate split.
# T=1 if avg(HR) in first 6 hours >= 90
t1, t0 = split_by_T(
    ts, ts_ids,
    mode="avg_value",
    var_name="HR",
    avg_value_threshold=90,
    start_minute=0,
    end_minute=360
)
print([len(elem) for elem in [t1, t0]])

# T=1 if avg time between HR measurements in first day <= 60 minutes
t1, t0 = split_by_T(
    ts, ts_ids,
    mode="time_spacing",
    var_name="HR",
    spacing_threshold=60,
    start_minute=0,
    end_minute=1440
)
print([len(elem) for elem in [t1, t0]])

# Both together (AND)
t1, t0 = split_by_T(
    ts, ts_ids,
    mode="both",
    var_name="HR",
    avg_value_threshold=90,
    spacing_threshold=60,
    start_minute=0,
    end_minute=1440
)
print([len(elem) for elem in [t1, t0]])
