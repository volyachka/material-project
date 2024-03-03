# first line: 103
@memory.cache
def load_data_slim(archive_file: str = "../data_Kahle2020/migrated.aiida") -> pd.DataFrame:
    df = load_data(archive_file=archive_file)
    return df[["stru_label", "stru_db", "stru_id", "stru_ase", "temp", "diff_mean", "diff_std", "diff_sem"]]
