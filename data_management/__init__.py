from typing import Optional
import json

import pandas as pd

from aiida_utils.load_data import load_data
from pymatgen.core import Structure


def make_csv(output_file: str, archive_file: Optional[str] = None):
    args = [archive_file] if archive_file else []
    data = load_data(*args)

    columns_of_interest = {
        "stru": "starting_structure",
        "traj": "trajectory",
        "temp": "temperature",
        "diff_mean": "diffusion_mean_cm2_s",
        "diff_std": "diffusion_std_cm2_s",
        "diff_sem": "diffusion_sem_cm2_s",
        "stru_label": "label",
        "stru_db": "src_database",
        "stru_id": "src_id"
    }
    data_slim = pd.DataFrame(
        {
            target_column: data[src_column]
            for src_column, target_column in columns_of_interest.items()
        }
    )
    data_slim["starting_structure"] = data_slim["starting_structure"].apply(lambda x: x.get_pymatgen().to_json())
    data_slim["first_frame_structure"] = data_slim["trajectory"].apply(lambda x: x.get_step_structure(0).get_pymatgen().to_json())
    data_slim["trajectory"] = data_slim["trajectory"].apply(lambda x: x.get_positions())
    data_slim.to_csv(output_file, index=False)

def read_csv(input_file: str) -> pd.DataFrame:
    data = pd.read_csv(input_file)
    data["starting_structure"] = data["starting_structure"].apply(lambda x: Structure.from_dict(json.loads(x)))
    data["first_frame_structure"] = data["first_frame_structure"].apply(lambda x: Structure.from_dict(json.loads(x)))

    return data
