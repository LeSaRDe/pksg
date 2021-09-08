#! /bin/bash

##################################################
# ENVIRONMENT VARIABLES
##################################################
# GLOBAL
DS_NAME="sample"
export DS_NAME

NUM_TASK=10
export NUM_TASK

JAVA_XMX="-Xmx5G"
SLURM_ARRAY_TASK_ID=0

# RAW_TXT_INFO
RAW_TXT_INFO_NUM_TASK=$NUM_TASK
export RAW_TXT_INFO_NUM_TASK
RAW_TXT_INFO_JOB_ID=0
export RAW_TXT_INFO_JOB_ID

# TXT_CLEAN
TXT_CLEAN_JOB_ID=0
export TXT_CLEAN_JOB_ID
TXT_CLEAN_NUM_TASK=$NUM_TASK
export TXT_CLEAN_NUM_TASK

# TXT_HASH
TXT_HASH_JOB_ID=0
export TXT_HASH_JOB_ID
TXT_HASH_NUM_TASK=$NUM_TASK
export TXT_HASH_NUM_TASK

# SEM_UNIT
SEM_UNIT_JOB_ID=0
export SEM_UNIT_JOB_ID
SEM_UNIT_NUM_TASK=$NUM_TASK
export SEM_UNIT_NUM_TASK
SEM_UNIT_CUST_PH_DS_NAME="covid"
export SEM_UNIT_CUST_PH_DS_NAME

# TXT_SENT
TXT_SENT_JOB_CNT=1
export TXT_SENT_JOB_CNT
TXT_SENT_NUM_TASK=$NUM_TASK
export TXT_SENT_NUM_TASK

# TXT_PH
TXT_PH_JOB_ID=0
export TXT_PH_JOB_ID
TXT_PH_NUM_TASK=$NUM_TASK
export TXT_PH_NUM_TASK

# SGRAPH
SGRAPH_NUM_TASK=$NUM_TASK
export SGRAPH_NUM_TASK

# PH_SENT
PH_SENT_NUM_TASK=$NUM_TASK
export PH_SENT_NUM_TASK

# PKSG
PKSG_NUM_TASK=$NUM_TASK
export PKSG_NUM_TASK

# T_INT
T_INT_INT_LEN="####5"
export T_INT_INT_LEN
T_INT_INT_STRIDE="####1"
export T_INT_INT_STRIDE

# SENT_TS
SENT_TS_PKSG_JOB_CNT=1
export SENT_TS_PKSG_JOB_CNT
SENT_TS_JOB_CNT=1
export SENT_TS_JOB_CNT
SENT_TS_NUM_TASK=$NUM_TASK
export SENT_TS_NUM_TASK
SENT_TS_QUERY_DS_NAME="covid_19_vaccination"
export SENT_TS_QUERY_DS_NAME
SENT_TS_QUOTIENT_NAME='covid_19_vaccination'
export SENT_TS_QUOTIENT_NAME
SENT_TS_PH_DS_NAME=$DS_NAME
export SENT_TS_PH_DS_NAME
SENT_TS_SHOW_IMG=0
export SENT_TS_SHOW_IMG
SENT_TS_SAVE_IMG=1
export SENT_TS_SAVE_IMG


##################################################
# PIPELINE STAGES
##################################################
# RAW_TW_INFO
echo "RAW_TXT Starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_raw_tw_info_$DS_NAME.py gen_tasks $RAW_TXT_INFO_NUM_TASK $RAW_TXT_INFO_JOB_ID
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_raw_tw_info_$DS_NAME.py raw_tw_info $RAW_TXT_INFO_NUM_TASK $RAW_TXT_INFO_JOB_ID $DS_NAME
echo "RAW_TXT is done!"

# TXT_CLEAN
echo "TXT_CLEAN starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_txt_clean_carley.py gen_tasks $TXT_CLEAN_NUM_TASK $TXT_CLEAN_JOB_ID $DS_NAME tw_raw_txt
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_txt_clean_carley.py txt_clean $TXT_CLEAN_NUM_TASK $TXT_CLEAN_JOB_ID $DS_NAME
echo "TXT_CLEAN is done!"

# TXT_HASH
echo "TXT_HASH starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_txt_hash.py gen_tasks $TXT_HASH_NUM_TASK $TXT_HASH_JOB_ID $DS_NAME
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_txt_hash.py txt_hash $TXT_HASH_NUM_TASK $TXT_HASH_JOB_ID $DS_NAME
echo "TXT_HASH is done!"

# SEM_UNIT
echo "SEM_UNIT starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_sem_unit.py gen_tasks $SEM_UNIT_NUM_TASK $SEM_UNIT_JOB_ID $DS_NAME
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_sem_unit.py sem_unit $SEM_UNIT_NUM_TASK $SEM_UNIT_JOB_ID $DS_NAME $SEM_UNIT_CUST_PH_DS_NAME
echo "SEM_UNIT is done!"

# TXT_SENT
echo "TXT_SENT gen_tasks starts."
python ../sentiment_analysis_and_forecasting/stanford_sentiments_preparation.py gen_tasks $TXT_SENT_NUM_TASK $TXT_SENT_JOB_CNT $DS_NAME
echo "TXT_SENT gen_tasks is done!"

echo "TXT SENT corenlp @ Job ${SLURM_ARRAY_TASK_ID} starts."
cd ../core_nlp_sentiments/
. envsetup.sh
make
java $JAVA_XMX core_nlp_sentiments.PhraseSentimentParallel $TXT_SENT_NUM_TASK ${SLURM_ARRAY_TASK_ID}
cd ./run_scripts/
echo "TXT SENT corenlp @ Job ${SLURM_ARRAY_TASK_ID} is done."

# TXT_PH
echo "TXT_PH starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_phrase.py gen_tasks $TXT_PH_NUM_TASK $TXT_PH_JOB_ID $DS_NAME
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_phrase.py phrase_ext $TXT_PH_NUM_TASK $TXT_PH_JOB_ID $DS_NAME
echo "TXT_PH is done!"

# SGRAPH
echo "SGRAPH starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_sgraph.py build_sgraph_from_json $SGRAPH_NUM_TASK ${SLURM_ARRAY_TASK_ID} $DS_NAME
echo "SGRAPH is done!"

# PH_SENT
echo "PH_SENT starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_phrase_sent.py gen_tasks $PH_SENT_NUM_TASK ${SLURM_ARRAY_TASK_ID} $DS_NAME
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_phrase_sent.py ph_sent $PH_SENT_NUM_TASK ${SLURM_ARRAY_TASK_ID} $DS_NAME
echo "PH_SENT is done!"

# PKSG
echo "PKSG starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_pksg.py gen_tasks $PKSG_NUM_TASK ${SLURM_ARRAY_TASK_ID} $DS_NAME
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_pksg.py pksg $PKSG_NUM_TASK ${SLURM_ARRAY_TASK_ID} $DS_NAME
echo "PKSG is done!"

# T_INT
echo "T_INT gen_int starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_time_series.py gen_time_series $DS_NAME $T_INT_INT_LEN $T_INT_INT_STRIDE
echo "T_INT gen_int is done!"

echo "T_INT pksg_ts starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_time_series.py pksg_ts ${SLURM_ARRAY_TASK_ID} $DS_NAME
echo "T_INT pksg_ts is done!"

# SENT_TS
echo "SENT_TS gen_tasks starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_sentiment_time_series.py gen_tasks $DS_NAME $SENT_TS_PKSG_JOB_CNT $SENT_TS_JOB_CNT $SENT_TS_NUM_TASK
echo "SENT_TS gen_tasks is done!"

echo "SENT_TS sent_ts starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_sentiment_time_series.py sent_ts $SENT_TS_NUM_TASK ${SLURM_ARRAY_TASK_ID} $DS_NAME $SENT_TS_QUERY_DS_NAME
echo "SENT_TS sent_ts is done!"

echo "SENT_TS merge_sent_ts starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_sentiment_time_series.py merge_sent_ts $DS_NAME $SENT_TS_QUERY_DS_NAME $SENT_TS_JOB_CNT $SENT_TS_QUOTIENT_NAME $SENT_TS_PH_DS_NAME
echo "SENT_TS merge_sent_ts is done!"

echo "SENT_TS draw_sent_ts starts."
python ../sentiment_analysis_and_forecasting/snt_ana_frcst_sentiment_time_series.py draw_sent_ts $DS_NAME $SENT_TS_QUERY_DS_NAME $SENT_TS_SHOW_IMG $SENT_TS_SAVE_IMG $SENT_TS_QUOTIENT_NAME
echo "SENT_TS draw_sent_ts is done!"