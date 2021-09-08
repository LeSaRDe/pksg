package core_nlp_sentiments;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;


/********************************************************************************
 * PKSG PIPELINE
 * STAGE: TEXT SENTIMENT (TXT_SENT)
 *
 * REQUIREMENTS:
 *     TXT_CLEAN
 *
 * OUTPUTS:
 *     text->sentiment tree table
 *     JSON
 *     Key: text id (String): G_TXT_IDX
 *     Value: sentiment tree string (JSON String)
 *
 * NOTE:
 *     The outputs are stored as intermediate JSON files instead of a merged JSON file.
 ********************************************************************************/


public class PhraseSentimentParallel {
    /********************************************************************************
     * Memeber Variables
     ********************************************************************************/
    private int m_num_tasks = 0;
    private String m_job_id = null;
    private ArrayList<PhraseSentimentTask> m_l_ps_task = null;
    private Logger m_logger = null;

    /********************************************************************************
     * Memeber Functions
     ********************************************************************************/
    public PhraseSentimentParallel(int num_task, String job_id) {
        m_num_tasks = num_task;
        m_job_id = job_id;
        m_l_ps_task = new ArrayList<>();
        m_logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
        m_logger.setLevel(GlobalSettings.LOG_LEVEL);
    }

    public void Go(String txt_idx_col, String clean_txt_col, String sgraph_col_name) {
        m_logger.log(Level.WARNING, "[Go] Starts.");
        for (int i=0; i<m_num_tasks; i++) {
            String task_id = m_job_id + "#" + i;
            PhraseSentimentTask ps_task = new PhraseSentimentTask(task_id, txt_idx_col, clean_txt_col, sgraph_col_name);
            m_l_ps_task.add(ps_task);
            ps_task.start();
        }
        int done_cnt = 0;
        while (m_l_ps_task.size() != done_cnt) {
            for (PhraseSentimentTask ps_task : m_l_ps_task) {
                if (ps_task.isAlive()) {
                    try {
                        ps_task.join(1);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                else {
                    done_cnt += 1;
                }
            }
        }
        m_logger.log(Level.WARNING, "[Go] All done.");
    }

    /********************************************************************************
     * Program Entry
     ********************************************************************************/
    public static void main(String[] args) {
        // 10 0
        int num_task = Integer.parseInt(args[0]);
        String job_id = args[1];
//        String txt_idx_col = null;
//        if (args.length >= 3) {
//            txt_idx_col = args[2];
//        }
//        String clean_txt_col = null;
//        if (args.length >= 4) {
//            clean_txt_col = args[3];
//        }
//        String sgraph_col_name = null;
//        if (args.length >= 5) {
//            sgraph_col_name = args[4];
//        }
        PhraseSentimentParallel ps_task_parallel = new PhraseSentimentParallel(num_task, job_id);
        ps_task_parallel.Go(GlobalSettings.G_TXT_IDX, GlobalSettings.G_CLEAN_TXT_COL, GlobalSettings.G_SGRAPH_COL);
    }
}
