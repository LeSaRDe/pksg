#! /bin/bash

##################################################
# ENVIRONMENT VARIABLES
##################################################
# GLOBAL
DS_NAME="va2021"
export DS_NAME

NUM_TASK=35
export NUM_TASK

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
TXT_SENT_JOB_CNT=10
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
T_INT_INT_LEN="1#3###"
export T_INT_INT_LEN
T_INT_INT_STRIDE="1####"
export T_INT_INT_STRIDE

# SENT_TS
SENT_TS_PKSG_JOB_CNT=10
export SENT_TS_PKSG_JOB_CNT
SENT_TS_JOB_CNT=10
export SENT_TS_JOB_CNT
SENT_TS_NUM_TASK=$NUM_TASK
export SENT_TS_NUM_TASK
SENT_TS_QUERY_DS_NAME="vaccination_intent"
export SENT_TS_QUERY_DS_NAME
SENT_TS_QUOTIENT_NAME='vaccination_intent'
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
#sbatch raw_tw_info.slurm
# TXT_CLEAN
#sbatch txt_clean.slurm
# TXT_HASH
#sbatch txt_hash.slurm
# SEM_UNIT
#sbatch sem_unit.slurm
# TXT_SENT
#sbatch txt_sent_gen_tasks.slurm
#sbatch --array=0-9 txt_sent_corenlp.slurm
# TXT_PH
#sbatch txt_ph.slurm
# SGRAPH
#sbatch --array=0-9 sgraph.slurm
# PH_SENT
#sbatch --array=0-9 ph_sent.slurm
# PKSG
#sbatch --array=0-9 pksg.slurm
# T_INT
#sbatch t_int_gen_int.slurm
#sbatch --array=0-9 t_int_pksg_ts.slurm
# SENT_TS
#sbatch sent_ts_gen_tasks.slurm
#sbatch --array=0-9 sent_ts_sent_ts.slurm
#sbatch sent_ts_merge_sent_ts.slurm
sbatch sent_ts_draw_sent_ts.slurm