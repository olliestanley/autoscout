import json
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Dict, Union

import pandas as pd


def sleep_and_return(result: Any, sleep_seconds: float) -> Any:
    sleep(sleep_seconds)
    return result


def load_json(
    file_path: Union[str, Path],
) -> Dict[str, Any]:
    file_path = Path(file_path)

    with open(file_path, "r") as f:
        loaded_json = json.load(f)

    return loaded_json


def load_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    file_path = Path(file_path)
    return pd.read_csv(file_path, **kwargs)


def write_dated_csv(
    df: pd.DataFrame,
    out_dir: Union[str, Path],
    basename: str,
    **kwargs,
) -> Path:
    out_dir = Path(out_dir)
    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
    out_path = out_dir / f"{basename}_{dt_string}.csv"
    df.to_csv(out_path, **kwargs)
    return out_path
