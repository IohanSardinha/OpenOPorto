from pathlib import Path
import json
import importlib.util
import pandas as pd
from sklearn.model_selection import train_test_split


def load_imob_using_processor(imob_dir: Path):
    # Try importing Processer as a package module so its relative imports work.
    repo_root = Path(__file__).resolve().parents[1]
    package_name = "Population.pipeline.oporto.IMob.Processer"
    try:
        import sys
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        mod = importlib.import_module(package_name)
        IMobProcesser = mod.IMobProcesser
    except Exception:
        # Fallback: load by absolute path (may fail if Processer uses relative imports)
        proc_path = repo_root / "Population" / "pipeline" / "oporto" / "IMob" / "Processer.py"
        spec = importlib.util.spec_from_file_location("imob_processer", str(proc_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        IMobProcesser = module.IMobProcesser

    # expected filenames in IMOB2017
    files = {
        "households": imob_dir / "TBL_alojamento_AMP.csv",
        "expenses": imob_dir / "TBL_alojamento_despesa_AMP.csv",
        "vehicles": imob_dir / "TBL_alojamento_veiculos_AMP.csv",
        "incomes": imob_dir / "TBL_alojamento_rendimentos_AMP.csv",
        "individuals": imob_dir / "TBL_individuos_AMP.csv",
        "passes": imob_dir / "TBL_tipo_de_passe_AMP.csv",
        "trips": imob_dir / "TBL_viagens_AMP.csv",
    }

    for k, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Required IMOB file missing: {p}")

    generic = IMobProcesser.read(
        files["households"],
        files["expenses"],
        files["vehicles"],
        files["incomes"],
        files["individuals"],
        files["passes"],
        files["trips"],
        fix_trips=True,
    )

    return generic


def generic_to_tables(generic: dict):
    persons = []
    trips = []

    for pid, data in generic.items():
        attrs = data.get("attributes", {})
        tripDesc = data.get("tripDesc", {})
        persons.append({
            "person_id": pid,
            "gender": attrs.get("gender"),
            "ageGroup": attrs.get("ageGroup"),
            "educationLvl": attrs.get("educationLvl"),
            "economicSituation": attrs.get("economicSituation"),
            "trip_type": tripDesc.get("type"),
            "trip_weekday": tripDesc.get("weekday"),
        })

        for ti, leg in enumerate(data.get("legs", []), start=1):
            trips.append({
                "person_id": pid,
                "trip_index": ti,
                "activity": leg.get("activity"),
                "distance": leg.get("distance"),
                "mode": leg.get("mode"),
                "departure": leg.get("departure"),
                "arrival": leg.get("arrival"),
            })

    persons_df = pd.DataFrame(persons)
    trips_df = pd.DataFrame(trips)
    return persons_df, trips_df


def save_outputs(out_dir: Path, persons_df: pd.DataFrame, trips_df: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prefer parquet when available, fallback to CSV to avoid extra dependencies
    try:
        persons_df.to_parquet(out_dir / "imob_persons.parquet", index=False)
        trips_df.to_parquet(out_dir / "imob_trips.parquet", index=False)
    except Exception:
        persons_df.to_csv(out_dir / "imob_persons.csv", index=False)
        trips_df.to_csv(out_dir / "imob_trips.csv", index=False)


def save_jsonl_by_person(out_dir: Path, generic: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "imob_persons.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for pid, data in generic.items():
            rec = {"person_id": pid, **data}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def train_test_split_persons(persons_df: pd.DataFrame, trips_df: pd.DataFrame, test_size=0.2, seed=42):
    ids = persons_df["person_id"].tolist()
    train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=seed)

    persons_train = persons_df[persons_df["person_id"].isin(train_ids)].reset_index(drop=True)
    persons_test = persons_df[persons_df["person_id"].isin(test_ids)].reset_index(drop=True)

    trips_train = trips_df[trips_df["person_id"].isin(train_ids)].reset_index(drop=True)
    trips_test = trips_df[trips_df["person_id"].isin(test_ids)].reset_index(drop=True)

    return (persons_train, trips_train), (persons_test, trips_test)


def run(imob_dir: str = None, out_dir: str = None, test_size: float = 0.2, seed: int = 42):
    base = Path(__file__).resolve().parents[1]
    imob_path = Path(imob_dir) if imob_dir else base / "Population" / ".data" / "IMOB2017"
    out_path = Path(out_dir) if out_dir else base / "data"

    generic = load_imob_using_processor(imob_path)
    persons_df, trips_df = generic_to_tables(generic)
    save_outputs(out_path, persons_df, trips_df)
    save_jsonl_by_person(out_path, generic)

    (persons_train, trips_train), (persons_test, trips_test) = train_test_split_persons(persons_df, trips_df, test_size=test_size, seed=seed)

    save_outputs(out_path / "train", persons_train, trips_train)
    save_outputs(out_path / "test", persons_test, trips_test)

    # also save jsonl splits for persona-focused LLM work
    # filter generic to sets
    train_jsonl = out_path / "train" / "imob_persons.jsonl"
    test_jsonl = out_path / "test" / "imob_persons.jsonl"
    with train_jsonl.open("w", encoding="utf-8") as tf, test_jsonl.open("w", encoding="utf-8") as ef:
        for pid, rec in generic.items():
            if pid in set(persons_train["person_id"]):
                tf.write(json.dumps({"person_id": pid, **rec}, ensure_ascii=False, default=str) + "\n")
            elif pid in set(persons_test["person_id"]):
                ef.write(json.dumps({"person_id": pid, **rec}, ensure_ascii=False, default=str) + "\n")

    print(f"Saved outputs under: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load IMOB2017, preprocess and split for TravelSurvey work")
    parser.add_argument("--imob", help="Path to IMOB2017 folder (default: ../Population/.data/IMOB2017)")
    parser.add_argument("--out", help="Output folder (default: TravelSurvey/data)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run(args.imob, args.out, args.test_size, args.seed)
