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

echo "RAW_TXT Starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_raw_tw_info_va2021.py gen_tasks $RAW_TXT_INFO_NUM_TASK $RAW_TXT_INFO_JOB_ID
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_raw_tw_info_va2021.py raw_tw_info $RAW_TXT_INFO_NUM_TASK $RAW_TXT_INFO_JOB_ID $DS_NAME
echo "RAW_TXT is done!"