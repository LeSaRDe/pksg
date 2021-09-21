package core_nlp_sentiments;

import java.util.logging.Level;
import java.nio.file.Paths;

public class GlobalSettings {
//    static final String WORK_DIR = "/scratch/mf3jh/data/covid/sent_graph/";
//     static final String WORK_DIR = "/home/mf3jh/workspace/data/aspect_sentiment_analysis/sent_graph/";
    static final String OUTPUT = System.getenv("OUTPUT");
    static final String WORK_DIR = Paths.get(OUTPUT, "sent_graph").toString(); //"D:\\workspace\\PKSG\\output\\data\\sent_graph\\";
    static final String INT_DIR = Paths.get(WORK_DIR, "int").toString();
    static final String SENT_TASK_FILE_FMT = Paths.get(INT_DIR, "txt_sent_task_%s.json").toString();
    static final String SENT_INT_FILE_FMT = Paths.get(INT_DIR, "txt_sent_int_%s.json").toString();
    static final Level LOG_LEVEL = Level.WARNING;
//     static final String G_TXT_IDX = "txt_id";
    static final String G_TXT_IDX = "hash_txt";
    static final String G_CLEAN_TXT_COL = "clean_txt";
    static final String G_SGRAPH_COL = "sgraph";
}
