package core_nlp_sentiments;

import javax.json.*;
import java.io.*;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;


public class PhraseSentimentTask extends Thread {
    /********************************************************************************
     * Inner Classes
     ********************************************************************************/
    private class SentTask {
        public String m_txt_id = null;
        public String m_clean_txt = null;

        public SentTask(String txt_id, String clean_txt) {
            m_txt_id = txt_id;
            m_clean_txt = clean_txt;
        }
    }

    /********************************************************************************
     * Memeber Variables
     ********************************************************************************/
    private CoreNLPSentiments m_corenlp_sentiments = null;
    private Logger m_logger = null;
    private ArrayList<SentTask> m_l_sent_task = null;
    private String m_task_id = null;
    private String m_txt_idx_col = "txt_id";
    private String m_sgraph_col_name = "txt_sgraph";

    /********************************************************************************
     * Memeber Functions
     ********************************************************************************/
    public PhraseSentimentTask(String task_id, String txt_idx_col, String clean_txt_col, String sgraph_col_name) {
        m_corenlp_sentiments = new CoreNLPSentiments();
        m_logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
        m_logger.setLevel(GlobalSettings.LOG_LEVEL);
        m_task_id = task_id;
        if (txt_idx_col != null) {
            m_txt_idx_col = txt_idx_col;
        }
        if (sgraph_col_name != null) {
            m_sgraph_col_name = sgraph_col_name;
        }
        m_l_sent_task = LoadSentTask(task_id, m_txt_idx_col, clean_txt_col);
    }

    private ArrayList<SentTask> LoadSentTask(String task_id, String txt_idx_col, String clean_txt_col) {
        InputStream in_fd = null;
        try {
            in_fd = new FileInputStream(String.format(GlobalSettings.SENT_TASK_FILE_FMT, task_id));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        JsonReader json_reader = Json.createReader(in_fd);
        JsonArray json_array = json_reader.readArray();
        json_reader.close();

        ArrayList<SentTask> l_sent_task = new ArrayList<>();
        for (int k=0; k<json_array.size(); k++) {
            JsonObject json_obj = json_array.getJsonObject(k);
            String txt_id = json_obj.getString(txt_idx_col);
            String clean_txt = json_obj.getString(clean_txt_col);
            l_sent_task.add(new SentTask(txt_id, clean_txt));
        }
        m_logger.log(Level.WARNING, String.format("[LoadSentTask] Thread %s: Load in %s tasks.",
                task_id, l_sent_task.size()));
        return l_sent_task;
    }

    @Override
    public void run() {
        Instant timer_start = Instant.now();
        Instant timer_end = null;
        Duration time_elapse = null;
        super.run();
        int done_cnt = 0;
        JsonArrayBuilder task_json_array_builder = Json.createArrayBuilder();
        for (SentTask sent_task : m_l_sent_task) {
            String txt_id = sent_task.m_txt_id;
            String clean_txt = sent_task.m_clean_txt;
            JsonArrayBuilder sgraph_array_builder = Json.createArrayBuilder();
            try {
                ArrayList<String> l_sgraph_json_str = m_corenlp_sentiments.text_to_sgraph_json_strs(clean_txt);
                if (l_sgraph_json_str != null && l_sgraph_json_str.size() > 0) {
                    for (String sgraph_json_str : l_sgraph_json_str) {
                        sgraph_array_builder.add(sgraph_json_str);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            JsonObjectBuilder txt_sent_json_out_builder = Json.createObjectBuilder();
            JsonObject txt_sent_json_out = txt_sent_json_out_builder
                    .add(m_txt_idx_col, txt_id)
                    .add(m_sgraph_col_name, sgraph_array_builder.build())
                    .build();
            task_json_array_builder.add(txt_sent_json_out);
            done_cnt += 1;
            if (done_cnt % 1000 == 0 && done_cnt >= 1000) {
                timer_end = Instant.now();
                time_elapse = Duration.between(timer_start, timer_end);
                m_logger.log(Level.WARNING, String.format("[run] Task %s: %d texts done in %d secs.",
                        m_task_id, done_cnt, time_elapse.getSeconds()));
            }
        }
        timer_end = Instant.now();
        time_elapse = Duration.between(timer_start, timer_end);
        m_logger.log(Level.WARNING, String.format("[run] Task %s: %d texts done in %d secs.",
                m_task_id, done_cnt, time_elapse.getSeconds()));

        JsonArray task_json_array = task_json_array_builder.build();
        try {
            FileWriter out_fd = new FileWriter(String.format(GlobalSettings.SENT_INT_FILE_FMT, m_task_id));
            JsonWriter json_writer = Json.createWriter(out_fd);
            json_writer.writeArray(task_json_array);
            json_writer.close();
            out_fd.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        timer_end = Instant.now();
        time_elapse = Duration.between(timer_start, timer_end);
        m_logger.log(Level.WARNING, String.format("[run] Task %s: All Done in %d secs.",
                m_task_id, time_elapse.getSeconds()));
    }
}
