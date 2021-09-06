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
module load java/1.12.0
source activate my_venv
conda deactivate
source activate my_venv

cd /home/mf3jh/workspace/core_nlp_sentiments/
. envsetup_uva.sh
#make
echo "TXT SENT corenlp @ Job ${SLURM_ARRAY_TASK_ID} starts."
java -Xmx300G core_nlp_sentiments.PhraseSentimentParallel $TXT_SENT_NUM_TASK ${SLURM_ARRAY_TASK_ID}
cd /home/mf3jh/workspace/pksg/
echo "TXT SENT corenlp @ Job ${SLURM_ARRAY_TASK_ID} is done."
