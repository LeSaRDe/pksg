pipeline {

    agent { label 'pksg' }

    environment {
        PYTHONPATH = "${WORKSPACE}"
        OUTPUT = "${WORKSPACE}/output"
        NUM_TASK = 10
        DS_NAME = "sample"
        LOCAL_OR_REMOTE = "local"
    }

    stages {

        stage("Setup ENV") {

          steps {
            echo "WORKSPACE ${WORKSPACE}"
            echo "OUTPUT ${OUTPUT}"
            sh 'cd $WORKSPACE'
        	sh 'chmod +x ./run_scripts/pksg_pipeline_envsetup.sh && ./run_scripts/pksg_pipeline_envsetup.sh'
          }

        }

        stage("RAW TXT INFO") {
            environment {
                RAW_TXT_INFO_JOB_ID = 0
                RAW_TXT_INFO_NUM_TASK = "${NUM_TASK}"
            }
            steps {
                echo "RAW_TXT Starts."
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_raw_tw_info_$DS_NAME.py gen_tasks $RAW_TXT_INFO_NUM_TASK $RAW_TXT_INFO_JOB_ID'
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_raw_tw_info_$DS_NAME.py raw_tw_info $RAW_TXT_INFO_NUM_TASK $RAW_TXT_INFO_JOB_ID $DS_NAME'
                echo "RAW_TXT is done!"
            }
        }

       stage("TXT CLEAN") {
            environment {
                TXT_CLEAN_JOB_ID = 0
                TXT_CLEAN_NUM_TASK = "${NUM_TASK}"
                RAW_TXT_COL = "tw_raw_txt"
            }
            steps {
                echo "TXT_CLEAN Starts."
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_txt_clean_carley.py gen_tasks $TXT_CLEAN_NUM_TASK $TXT_CLEAN_JOB_ID $DS_NAME $RAW_TXT_COL'
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_txt_clean_carley.py raw_tw_info $TXT_CLEAN_NUM_TASK $TXT_CLEAN_JOB_ID $DS_NAME'
                echo "TXT_CLEAN is done!"
            }
        }
    }
}
