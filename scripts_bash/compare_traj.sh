#!/bin/bash
if [ $1 == "withoffset" ]; then
    echo "Comparing with offset"
    PLANNED="human_data/open_drawer/traj_1/processed_3d/replayable_traj_withoffset.npy"
    EPISODE_PATH="data/eval/2026-01-13/2026-01-13_23-35-45"
else
    echo "Comparing without offset"
    PLANNED="human_data/open_drawer/traj_1/processed_3d/replayable_traj_withoutoffset.npy"
    EPISODE_PATH="data/eval/2026-01-13/2026-01-13_23-15-53"
fi
EXECUTED=$EPISODE_PATH"/trajectory.npz"

python scripts/compare_traj.py --planned $PLANNED --executed $EXECUTED