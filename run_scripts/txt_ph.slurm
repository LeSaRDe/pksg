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

echo "TXT_PH starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_phrase.py gen_tasks $TXT_PH_NUM_TASK $TXT_PH_JOB_ID $DS_NAME
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_phrase.py phrase_ext $TXT_PH_NUM_TASK $TXT_PH_JOB_ID $DS_NAME
echo "TXT_PH is done!"
