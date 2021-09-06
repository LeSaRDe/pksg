#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=350000
#SBATCH --partition=bii-gpu
#SBATCH --exclusive
#SBATCH --cpus-per-task=20
#SBTACH --core-spec=40
#SBATCH --gres=gpu:v100:4

module load anaconda
source activate my_venv
conda deactivate
source activate my_venv

echo "SGRAPH starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_sgraph.py build_sgraph_from_json $SGRAPH_NUM_TASK ${SLURM_ARRAY_TASK_ID} $DS_NAME
echo "SGRAPH is done!"
