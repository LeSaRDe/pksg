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

echo "TXT_SENT gen_tasks starts."
python ../sentiment_analysis_and_forecasting/stanford_sentiments_preparation.py gen_tasks $TXT_SENT_NUM_TASK $TXT_SENT_JOB_CNT $DS_NAME
echo "TXT_SENT gen_tasks is done!"