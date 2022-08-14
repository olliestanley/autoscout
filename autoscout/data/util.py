import datetime
from pathlib import Path
from typing import Union

import pandas as pd


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
