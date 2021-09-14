import logging
import time
import json
import math
import sys

import pandas as pd

from util import global_settings


def gen_sentiment_tasks(ds_name, num_tasks, job_cnt, txt_idx_col=global_settings.g_hash_txt_col,
                        clean_txt_col=global_settings.g_clean_txt_col):
    """
    'job_cnt' provides the total number of jobs. 'num_tasks' provides the number of tasks in each job.
    """
    logging.debug('[gen_sentiment_tasks] Starts.')
    timer_start = time.time()

    num_tasks = int(num_tasks)
    job_cnt = int(job_cnt)
    if num_tasks <= 0 or job_cnt <= 0:
        raise Exception('[gen_sentiment_tasks] Invalid num_tasks %s or job_cnt %s' % (num_tasks, job_cnt))

    df_hash_txt = pd.read_pickle(global_settings.g_txt_hash_file_fmt.format(ds_name))
    # df_hash_txt = df_hash_txt.loc[df_hash_txt[clean_txt_col].notnull()]
    num_clean_txt = len(df_hash_txt)
    logging.debug('[gen_sentiment_tasks] Load in %s clean texts.' % str(num_clean_txt))


    total_num_tasks = num_tasks * job_cnt
    batch_size = math.ceil(num_clean_txt / total_num_tasks)
    if batch_size < 1:
        batch_size = 1
    l_tasks = []
    for i in range(0, num_clean_txt, batch_size):
        if i + batch_size < num_clean_txt:
            l_tasks.append(df_hash_txt.iloc[i:i + batch_size])
        else:
            l_tasks.append(df_hash_txt.iloc[i:])
    logging.debug('[gen_sentiment_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    is_done = False
    for job_id in range(job_cnt):
        for task_id in range(num_tasks):
            abs_task_id = job_id * num_tasks + task_id
            if abs_task_id >= len(l_tasks):
                is_done = True
                break
            df_task = l_tasks[abs_task_id]
            l_json_out = []
            for hash_txt_key, clean_txt_rec in df_task.iterrows():
                clean_txt = clean_txt_rec[clean_txt_col]
                l_json_out.append({txt_idx_col: str(hash_txt_key), clean_txt_col: clean_txt})
            with open(global_settings.g_txt_sent_task_file_fmt.format(str(job_id) + '#' + str(task_id)), 'w+') as out_fd:
                json.dump(l_json_out, out_fd)
                out_fd.close()
        if is_done:
            break
    logging.debug('[gen_sentiment_tasks] All done with %s tasks in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        # gen_tasks 10 1 va2021
        num_tasks = sys.argv[2]
        job_cnt = sys.argv[3]
        ds_name = sys.argv[4]
        gen_sentiment_tasks(ds_name, num_tasks, job_cnt)
