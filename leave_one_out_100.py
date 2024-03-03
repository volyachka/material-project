from utils import *
from scoring import * 

if __name__ == "__main__":
    dfs = []
    for i in range(5):
        dfs.append(load_csv(f"groups_and_oxi_states_5_frames/df_features_with_barrier_step_{i}.pkl"))
    preds_without_traj, preds_with_traj, y = test_function_leave_one_out_100(dfs)

    with open("results/leave_one_out_labels_100.txt", "w") as output:
        output.write(str(y))
