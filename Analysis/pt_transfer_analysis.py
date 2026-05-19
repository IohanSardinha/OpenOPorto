from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


SECONDS_IN_DAY = 24 * 3600
DEFAULT_WALK_SPEED_KMH = 4.0
DEFAULT_WALK_SPEED_MPS = DEFAULT_WALK_SPEED_KMH * 1000.0 / 3600.0


@dataclass
class TransferAnalysisResult:
    transfers: pd.DataFrame
    person_summary: pd.DataFrame
    trip_summary: pd.DataFrame
    global_summary: dict


def matsim_time_to_seconds(value: object) -> float:
    """Convert MATSim time formats (HH:MM:SS or numeric seconds) to seconds."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    text = str(value).strip()
    if text == "":
        return np.nan

    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return np.nan

    parts = text.split(":")
    if len(parts) == 2:
        hours, minutes = parts
        seconds = "0"
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        return np.nan

    try:
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    except ValueError:
        return np.nan


def seconds_to_hhmmss(seconds: float) -> str:
    """Format seconds as HH:MM:SS, supporting durations over 24h."""
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return ""

    total = int(round(seconds))
    sign = "-" if total < 0 else ""
    total = abs(total)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{sign}{h:02d}:{m:02d}:{s:02d}"


def add_realistic_transfer_estimates(
    transfers_df: pd.DataFrame,
    *,
    walk_speed_kmh: float = DEFAULT_WALK_SPEED_KMH,
) -> pd.DataFrame:
    """Add estimated walk and wait times using a constant walk speed."""
    result = transfers_df.copy()
    walk_speed_mps = walk_speed_kmh * 1000.0 / 3600.0

    result["estimated_walk_speed_kmh"] = walk_speed_kmh
    result["estimated_walk_time_s"] = pd.to_numeric(result["transfer_distance_m"], errors="coerce").fillna(0.0) / walk_speed_mps
    result["estimated_wait_time_s"] = np.where(
        result["transfer_time_s"].notna(),
        np.maximum(0.0, result["transfer_time_s"] - result["estimated_walk_time_s"]),
        np.nan,
    )
    return result


def load_legs(legs_csv_path: str | Path) -> pd.DataFrame:
    """Load MATSim output_legs.csv(.gz) and add parsed numeric columns."""
    legs_csv_path = Path(legs_csv_path)
    df = pd.read_csv(legs_csv_path, sep=";")

    required = {"person", "trip_id", "dep_time", "trav_time", "distance", "mode"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in legs file: {sorted(missing)}")

    df["dep_time_sec"] = df["dep_time"].map(matsim_time_to_seconds)
    df["trav_time_sec"] = df["trav_time"].map(matsim_time_to_seconds)
    if "wait_time" in df.columns:
        df["wait_time_sec"] = df["wait_time"].map(matsim_time_to_seconds)
    else:
        df["wait_time_sec"] = np.nan

    df["distance_m"] = pd.to_numeric(df["distance"], errors="coerce")
    df["_row"] = np.arange(len(df))

    # Stable sort ensures original leg order is preserved when departure times tie.
    df = df.sort_values(["person", "trip_id", "dep_time_sec", "_row"], kind="mergesort")
    return df


def extract_pt_transfers(
    legs_df: pd.DataFrame,
    *,
    pt_mode: str = "pt",
    transfer_modes: Optional[Iterable[str]] = ("walk",),
) -> pd.DataFrame:
    """
    Extract transfer episodes between consecutive PT legs within each person trip.

    transfer_time_s is measured from previous PT arrival to next PT departure.
    transfer_distance_m and transfer_walk_time_s sum selected non-PT legs between PT legs.
    """
    mode_filter = None if transfer_modes is None else set(transfer_modes)

    records: list[dict] = []

    for (person, trip_id), trip_legs in legs_df.groupby(["person", "trip_id"], sort=False):
        trip_legs = trip_legs.reset_index(drop=True)
        pt_positions = trip_legs.index[trip_legs["mode"] == pt_mode].tolist()

        if len(pt_positions) < 2:
            continue

        for transfer_index, (prev_i, next_i) in enumerate(zip(pt_positions[:-1], pt_positions[1:]), start=1):
            #if next_i - prev_i != 2:
            #        continue
            
            prev_pt = trip_legs.loc[prev_i]
            next_pt = trip_legs.loc[next_i]
            between = trip_legs.iloc[prev_i + 1 : next_i]

            if between.empty:
                considered = between
            else:
                considered = between[between["mode"] != pt_mode]
                if mode_filter is not None:
                    considered = considered[considered["mode"].isin(mode_filter)]

            prev_arrival = prev_pt["dep_time_sec"] + prev_pt["trav_time_sec"]
            next_departure = next_pt["dep_time_sec"]
            transfer_time_s = next_departure - prev_arrival
            if pd.notna(transfer_time_s) and transfer_time_s < 0:
                transfer_time_s += SECONDS_IN_DAY

            transfer_walk_time_s = considered["trav_time_sec"].fillna(0).sum()
            transfer_distance_m = considered["distance_m"].fillna(0).sum()
            transfer_wait_time_s = np.nan
            if pd.notna(transfer_time_s):
                transfer_wait_time_s = max(0.0, transfer_time_s - transfer_walk_time_s)

            records.append(
                {
                    "person": person,
                    "trip_id": trip_id,
                    "pt_legs_in_trip": len(pt_positions),
                    "transfer_index": transfer_index,
                    "prev_pt_dep_time": prev_pt["dep_time"],
                    "prev_pt_arrival_time": seconds_to_hhmmss(prev_arrival),
                    "next_pt_dep_time": next_pt["dep_time"],
                    "transfer_time_s": transfer_time_s,
                    "transfer_walk_time_s": transfer_walk_time_s,
                    "transfer_wait_time_s": transfer_wait_time_s,
                    "transfer_distance_m": transfer_distance_m,
                    "legs_between_count": len(between),
                    "modes_between": ",".join(between["mode"].fillna("")) if len(between) else "",
                    "prev_transit_line": prev_pt.get("transit_line", np.nan),
                    "prev_transit_route": prev_pt.get("transit_route", np.nan),
                    "next_transit_line": next_pt.get("transit_line", np.nan),
                    "next_transit_route": next_pt.get("transit_route", np.nan),
                    "prev_egress_stop_id": prev_pt.get("egress_stop_id", np.nan),
                    "next_access_stop_id": next_pt.get("access_stop_id", np.nan),
                }
            )

    return pd.DataFrame(records)


def summarize_transfer_stats(transfers_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build person-level, trip-level and global summaries from transfer episodes."""
    if transfers_df.empty:
        empty_person = pd.DataFrame(
            columns=[
                "person",
                "n_trips_with_transfers",
                "n_transfers",
                "total_transfer_time_s",
                "total_transfer_wait_time_s",
                "total_transfer_walk_time_s",
                "total_transfer_distance_m",
                "mean_transfer_time_s",
                "mean_transfer_distance_m",
            ]
        )
        empty_trip = pd.DataFrame(
            columns=[
                "person",
                "trip_id",
                "n_transfers",
                "trip_transfer_time_s",
                "trip_transfer_wait_time_s",
                "trip_transfer_walk_time_s",
                "trip_transfer_distance_m",
            ]
        )
        return (
            empty_person,
            empty_trip,
            {
                "n_agents_with_transfers": 0,
                "n_trips_with_transfers": 0,
                "n_transfer_events": 0,
                "mean_transfer_time_s": np.nan,
                "median_transfer_time_s": np.nan,
                "p95_transfer_time_s": np.nan,
                "mean_transfer_distance_m": np.nan,
                "median_transfer_distance_m": np.nan,
                "p95_transfer_distance_m": np.nan,
                "total_transfer_time_s": 0.0,
                "total_transfer_distance_m": 0.0,
            },
        )

    trip_summary = (
        transfers_df.groupby(["person", "trip_id"], as_index=False)
        .agg(
            n_transfers=("transfer_index", "count"),
            trip_transfer_time_s=("transfer_time_s", "sum"),
            trip_transfer_wait_time_s=("transfer_wait_time_s", "sum"),
            trip_transfer_walk_time_s=("transfer_walk_time_s", "sum"),
            trip_transfer_distance_m=("transfer_distance_m", "sum"),
        )
        .sort_values(["n_transfers", "trip_transfer_time_s"], ascending=[False, False])
    )

    person_summary = (
        transfers_df.groupby("person", as_index=False)
        .agg(
            n_trips_with_transfers=("trip_id", "nunique"),
            n_transfers=("transfer_index", "count"),
            total_transfer_time_s=("transfer_time_s", "sum"),
            total_transfer_wait_time_s=("transfer_wait_time_s", "sum"),
            total_transfer_walk_time_s=("transfer_walk_time_s", "sum"),
            total_transfer_distance_m=("transfer_distance_m", "sum"),
            mean_transfer_time_s=("transfer_time_s", "mean"),
            mean_transfer_distance_m=("transfer_distance_m", "mean"),
        )
        .sort_values(["n_transfers", "total_transfer_time_s"], ascending=[False, False])
    )

    global_summary = {
        "n_agents_with_transfers": int(transfers_df["person"].nunique()),
        "n_trips_with_transfers": int(transfers_df[["person", "trip_id"]].drop_duplicates().shape[0]),
        "n_transfer_events": int(len(transfers_df)),
        "mean_transfer_time_s": float(transfers_df["transfer_time_s"].mean()),
        "median_transfer_time_s": float(transfers_df["transfer_time_s"].median()),
        "p95_transfer_time_s": float(transfers_df["transfer_time_s"].quantile(0.95)),
        "mean_transfer_distance_m": float(transfers_df["transfer_distance_m"].mean()),
        "median_transfer_distance_m": float(transfers_df["transfer_distance_m"].median()),
        "p95_transfer_distance_m": float(transfers_df["transfer_distance_m"].quantile(0.95)),
        "total_transfer_time_s": float(transfers_df["transfer_time_s"].sum()),
        "total_transfer_distance_m": float(transfers_df["transfer_distance_m"].sum()),
    }

    return person_summary, trip_summary, global_summary


def analyze_pt_transfer_synchronism(
    legs_csv_path: str | Path,
    *,
    pt_mode: str = "pt",
    transfer_modes: Optional[Iterable[str]] = ("walk",),
) -> TransferAnalysisResult:
    """Run end-to-end PT transfer synchronism analysis from MATSim legs output."""
    legs = load_legs(legs_csv_path)
    transfers = extract_pt_transfers(legs, pt_mode=pt_mode, transfer_modes=transfer_modes)
    transfers = add_realistic_transfer_estimates(transfers)
    person_summary, trip_summary, global_summary = summarize_transfer_stats(transfers)
    return TransferAnalysisResult(
        transfers=transfers,
        person_summary=person_summary,
        trip_summary=trip_summary,
        global_summary=global_summary,
    )
