package core_nlp_sentiments;

import java.util.logging.Level;

public class GlobalSettings {
//    static final String WORK_DIR = "/scratch/mf3jh/data/covid/sent_graph/";
//     static final String WORK_DIR = "/home/mf3jh/workspace/data/aspect_sentiment_analysis/sent_graph/";
   static final String WORK_DIR = "D:\\workspace\\PKSG\\output\\data\\sent_graph\\";
    static final String INT_DIR = WORK_DIR + "int/";
    static final String SENT_TASK_FILE_FMT = INT_DIR + "txt_sent_task_%s.json";
    static final String SENT_INT_FILE_FMT = INT_DIR + "txt_sent_int_%s.json";
    static final Level LOG_LEVEL = Level.WARNING;
//     static final String G_TXT_IDX = "txt_id";
    static final String G_TXT_IDX = "hash_txt";
    static final String G_CLEAN_TXT_COL = "clean_txt";
    static final String G_SGRAPH_COL = "sgraph";
}
