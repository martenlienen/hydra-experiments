import pickle
from pathlib import Path

import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


def save_result(config, result):
    if config.output_path is None:
        return

    output_path = Path(to_absolute_path(config.output_path))
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with output_path.open("wb") as f:
        pickle.dump({"config": config, "result": result}, f)


def load_results(path):
    root = Path(path)
    rows = []
    for file in root.glob("*.pickle"):
        with file.open("rb") as f:
            data = pickle.load(f)
        cfg = OmegaConf.to_container(data["config"], resolve=True)
        result = data["result"]

        row = pd.json_normalize(cfg)
        row["f(x)"] = result.fun
        rows.append(row)
    df = pd.concat(rows)
    return df
