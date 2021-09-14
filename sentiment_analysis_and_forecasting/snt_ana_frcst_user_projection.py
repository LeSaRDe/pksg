import logging
import math
import sys
import time

import pandas as pd

from util import global_settings


def gen_rtug_ts_tasks(num_task, job_id, t_int_ds_name):
    logging.debug('[gen_rtug_ts_tasks] Starts.')
    timer_start = time.time()

    df_t_int = pd.read_pickle(global_settings.g_t_int_file_fmt.format(t_int_ds_name))
    num_t_int = len(df_t_int)
    logging.debug('[gen_rtug_ts_tasks] Load in %s time intervals.' % str(num_t_int))

    num_task = int(num_task)
    batch_size = math.ceil(num_t_int / num_task)
    l_tasks = []
    for i in range(0, num_t_int, batch_size):
        if i + batch_size < num_t_int:
            l_tasks.append(df_t_int.iloc[i:i + batch_size])
        else:
            l_tasks.append(df_t_int.iloc[i:])

    for idx, task in enumerate(l_tasks):
        task_id = str(job_id) + '#' + str(idx)
        pd.to_pickle(task, global_settings.g_rtug_ts_task_file_fmt.format(task_id))
    logging.debug('[gen_rtug_ts_tasks] %s rtug tasks are ready in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))


def rtug_single_task(task_id, df_raw_tw_info):
    logging.debug('[rtug_single_task] Tasks %s: Starts.')
    timer_start = time.time()

    df_task = pd.read_pickle(global_settings.g_rtug_ts_task_file_fmt.format(task_id))
    logging.debug('[rtug_single_task] Task %s: Load in %s t_int.' % (task_id, len(df_task)))

    for t_int_id, t_int_rec in df_task.iterrows():
        t_int_start = t_int_rec[global_settings.g_t_int_start_col]
        t_int_end = t_int_rec[global_settings.g_t_int_end_col]
        l_txt_ids = t_int_rec[global_settings.g_t_int_txt_ids]
        rtug = rtug_for_one_t_int(t_int_start, t_int_end, l_txt_ids, df_raw_tw_info)


def rtug_for_one_t_int(t_int_start, t_int_end, l_txt_ids, df_raw_tw_info):
    return








if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        # gen_tasks 10 0 202005
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        t_int_ds_name = sys.argv[4]
        gen_rtug_ts_tasks(num_task, job_id, t_int_ds_name)
