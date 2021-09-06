#!/bin/bash

#SBATCH --time=3-00:00:00
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

echo "SEM_UNIT starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_sem_unit.py gen_tasks $SEM_UNIT_NUM_TASK $SEM_UNIT_JOB_ID $DS_NAME
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_sem_unit.py sem_unit $SEM_UNIT_NUM_TASK $SEM_UNIT_JOB_ID $DS_NAME $SEM_UNIT_CUST_PH_DS_NAME
echo "SEM_UNIT is done!"