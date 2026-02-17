#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate polymetis-local
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
python controller.py