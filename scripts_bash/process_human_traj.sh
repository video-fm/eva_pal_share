#!/bin/bash

TASK="open_cabinet"
TRAJ=0

# python -m eva.utils.human_traj_adapter --task $TASK --traj $TRAJ --no-offset
python -m eva.utils.human_traj_adapter --task $TASK --traj $TRAJ 