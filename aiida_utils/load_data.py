from typing import List, Tuple

import pandas as pd
from aiida.storage.sqlite_zip.backend import SqliteZipBackend
from aiida import load_profile, get_profile
from aiida.orm import Group, Node, QueryBuilder, User

from . import find_connections

def load_data(archive_file: str = "../data_Kahle2020/migrated.aiida") -> List[Tuple[Node, Node, Node]]:
    load_profile(SqliteZipBackend.create_profile(archive_file))

    # Fixing missing user, as in https://aiida.discourse.group/t/setting-up-a-user-for-sqlitezipbackend/139/3
    (user,) = QueryBuilder().append(User).all(flat=True)
    get_profile().default_user_email = user.email

    traj_group = Group.objects.get(label="concatenated_trajectories")
    diff_group = Group.objects.get(label="diffusion_coefficients")
    stru_group = Group.objects.get(label="starting_structures")

    connections_traj_diff_l2r = find_connections(traj_group.nodes, diff_group.nodes, mode="from_left")
    connections_traj_diff_r2l = find_connections(traj_group.nodes, diff_group.nodes, mode="to_right")

    connections_stru_traj_l2r = find_connections(stru_group.nodes, traj_group.nodes, mode="from_left")
    connections_stru_traj_r2l = find_connections(stru_group.nodes, traj_group.nodes, mode="to_right")

    # There's a 1-to-1 correspondence between trajectories and diffusion results:
    assert connections_traj_diff_l2r == connections_traj_diff_r2l
    for l, r in connections_traj_diff_l2r:
        assert len(l) == 1
        assert len(r) == 1

    # So we can define:
    connections_traj_diff = [(l[0], r[0]) for l, r in connections_traj_diff_l2r]

    # But that's not the case for starting structures
    assert connections_stru_traj_l2r != connections_stru_traj_r2l

    # Actually, there's a single initial structure for every trajectory:
    for l, r in connections_stru_traj_r2l:
        assert len(l) == 1
        assert len(r) == 1

    # So we can define:
    connections_stru_traj = [(l[0], r[0]) for l, r in connections_stru_traj_r2l]

    # Now we can connect the data into triplets (stucture, trajectory, diffusion)
    assert len(connections_stru_traj) == len(connections_traj_diff)
    for (_, t1), (t2, _) in zip(connections_stru_traj, connections_traj_diff):
        assert t1 == t2

    connections_stru_traj_diff = [
        (s, t, d) for (s, t), (_, d) in zip(connections_stru_traj, connections_traj_diff)
    ]
    return connections_stru_traj_diff

def load_data_pd(archive_file: str = "../data_Kahle2020/migrated.aiida") -> pd.DataFrame:
    return pd.DataFrame(
        load_data(archive_file), columns=["stru", "traj", "diff"],
    )
