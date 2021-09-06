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

echo "SENT_TS merge_sent_ts starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_sentiment_time_series.py merge_sent_ts $DS_NAME $SENT_TS_QUERY_DS_NAME $SENT_TS_JOB_CNT $SENT_TS_QUOTIENT_NAME $SENT_TS_PH_DS_NAME
echo "SENT_TS merge_sent_ts is done!"
