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
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_txt_clean_carley.py txt_clean $TXT_CLEAN_NUM_TASK $TXT_CLEAN_JOB_ID $DS_NAME'
                echo "TXT_CLEAN is done!"
            }
        }

        stage("TXT HASH") {
            environment {
                TXT_HASH_JOB_ID = 0
                TXT_HASH_NUM_TASK = "${NUM_TASK}"
            }
            steps {
                echo "TXT_HASH Starts."
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_txt_hash.py gen_tasks $TXT_HASH_NUM_TASK $TXT_HASH_JOB_ID $DS_NAME'
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_txt_hash.py txt_hash $TXT_HASH_NUM_TASK $TXT_HASH_JOB_ID $DS_NAME'
                echo "TXT_HASH is done!"
            }
        }

        stage("SEMANTIC UNIT") {
            environment {
                SEM_UNIT_JOB_ID = 0
                SEM_UNIT_NUM_TASK = "${NUM_TASK}"
                SEM_UNIT_CUST_PH_DS_NAME = "covid"
            }
            steps {
                echo "SEM_UNIT Starts."
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_sem_unit.py gen_tasks $SEM_UNIT_NUM_TASK $SEM_UNIT_JOB_ID $DS_NAME'
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_sem_unit.py sem_unit $SEM_UNIT_NUM_TASK $SEM_UNIT_JOB_ID $DS_NAME $SEM_UNIT_CUST_PH_DS_NAME'
                echo "SEM_UNIT is done!"
            }
        }

        stage("TXT SENT") {
            environment {
                TXT_SENT_JOB_CNT = 1
                TXT_SENT_NUM_TASK = "${NUM_TASK}"
                SLURM_ARRAY_TASK_ID = 0
                JAVA_XMX = "-Xmx5G"
                CORENLP_PATH = "${CORENLP_ROOT}/${CORENLP_VER}/"
                _JAVA_OPTIONS = "-Djava.net.preferIPv4Stack=true"
                CLASSPATH = sh(returnStdout: true, script: '''#!/bin/bash
                        CLASSPATH="$CLASSPATH:.:./bin/:./config/"
                        for file in `find $CORENLP_ROOT/$CORENLP_VER/  -name "*.jar"`; do CLASSPATH="$CLASSPATH:`realpath $file`"; done
                        echo "$CLASSPATH"''')
            }

            steps {
                echo "TXT_SENT gen_tasks starts."
                sh 'python3 sentiment_analysis_and_forecasting/stanford_sentiments_preparation.py gen_tasks $TXT_SENT_NUM_TASK $TXT_SENT_JOB_CNT $DS_NAME'
                echo "TXT_SENT gen_tasks is done!"

                echo "TXT SENT corenlp @ Job ${SLURM_ARRAY_TASK_ID} starts."
                dir('core_nlp_sentiments') {
                    sh 'make'
                    sh 'java $JAVA_XMX -cp $CLASSPATH core_nlp_sentiments.PhraseSentimentParallel $TXT_SENT_NUM_TASK $SLURM_ARRAY_TASK_ID'
                }
                echo "TXT SENT corenlp @ Job ${SLURM_ARRAY_TASK_ID} is done."
            }
        }

        stage("TXT PHRASE") {
            environment {
                TXT_PH_JOB_ID = 0
                TXT_PH_NUM_TASK = "${NUM_TASK}"
            }
            steps {
                echo "TXT_PH Starts."
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_phrase.py gen_tasks $TXT_PH_NUM_TASK $TXT_PH_JOB_ID $DS_NAME'
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_phrase.py phrase_ext $TXT_PH_NUM_TASK $TXT_PH_JOB_ID $DS_NAME'
                echo "TXT_PH is done!"
            }
        }

        stage("SGRAPH") {
            environment {
                SGRAPH_NUM_TASK = "${NUM_TASK}"
            }
            steps {
                echo "SGRAPH Starts."
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_sgraph.py build_sgraph_from_json $SGRAPH_NUM_TASK $SLURM_ARRAY_TASK_ID $DS_NAME'
                echo "SGRAPH is done!"
            }
        }

        stage("PHRASE SENT") {
            environment {
                PH_SENT_NUM_TASK = "${NUM_TASK}"
            }
            steps {
                echo "PH_SENT Starts."
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_phrase_sent.py gen_tasks $PH_SENT_NUM_TASK $SLURM_ARRAY_TASK_ID $DS_NAME'
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_phrase_sent.py ph_sent $PH_SENT_NUM_TASK $SLURM_ARRAY_TASK_ID $DS_NAME'
                echo "PH_SENT is done!"
            }
        }

        stage("PKSG") {
            environment {
                PKSG_NUM_TASK = "${NUM_TASK}"
            }
            steps {
                echo "PKSG Starts."
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_pksg.py gen_tasks $PKSG_NUM_TASK $SLURM_ARRAY_TASK_ID $DS_NAME'
                sh 'python3 sentiment_analysis_and_forecasting/snt_ana_frcst_pksg.py pksg $PKSG_NUM_TASK $SLURM_ARRAY_TASK_ID $DS_NAME'
                echo "PKSG is done!"
            }
        }
    }
}
