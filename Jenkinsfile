pipeline {

    agent { label 'pksg' }

    environment {
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
            echo "Create venv"
            sh 'python3 -m venv venv'
            echo "Activate venv"
            sh '. venv/bin/activate'
            echo "Install package"
            sh 'pip install -e .'
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
                sh 'python3 ./sentiment_analysis_and_forecasting/snt_ana_frcst_raw_tw_info_$DS_NAME.py gen_tasks $RAW_TXT_INFO_NUM_TASK $RAW_TXT_INFO_JOB_ID'
                sh 'python3 ./sentiment_analysis_and_forecasting/snt_ana_frcst_raw_tw_info_$DS_NAME.py raw_tw_info $RAW_TXT_INFO_NUM_TASK $RAW_TXT_INFO_JOB_ID $DS_NAME'
                echo "RAW_TXT is done!"
            }
        }
    }
}
