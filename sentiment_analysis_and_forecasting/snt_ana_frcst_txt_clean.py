import logging
import sys
import math
import pandas as pd

import scenario_settings
import global_settings
from txt_cleansing import txt_clean_wrapper


def gen_txt_clean_tasks(ds_name, num_task, job_id, raw_txt_col):
    '''
    Task format:
        pandas DataFrame
        index: text id (str)
        columns: raw text (str)
    '''
    logging.debug('[gen_txt_clean_tasks] Starts.')

    df_raw_tw_info = pd.read_pickle(global_settings.g_raw_txt_info_file_fmt.format(ds_name))
    logging.debug('[gen_txt_clean_tasks] Load in %s tasks.' % len(df_raw_tw_info))

    df_raw_txt = df_raw_tw_info[[raw_txt_col]]
    df_raw_txt = df_raw_txt.rename(columns={raw_txt_col: global_settings.g_raw_txt_col})
    df_raw_txt = df_raw_txt.rename_axis(global_settings.g_txt_idx)
    num_txt = len(df_raw_txt)

    num_task = int(num_task)
    batch_size = math.ceil(num_txt / num_task)
    l_tasks = []
    for i in range(0, num_txt, batch_size):
        if i + batch_size < num_txt:
            l_tasks.append(df_raw_txt.iloc[i:i + batch_size])
        else:
            l_tasks.append(df_raw_txt.iloc[i:])

    for idx, task in enumerate(l_tasks):
        task_id = str(job_id) + '#' + str(idx)
        pd.to_pickle(task, global_settings.g_txt_clean_task_file_fmt.format(task_id))
    logging.debug('[gen_txt_clean_tasks] %s txt clean tasks are ready.' % len(l_tasks))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        # gen_tasks 10 0 va2021 tw_raw_txt
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        raw_txt_col = sys.argv[5]
        gen_txt_clean_tasks(ds_name, num_task, job_id, raw_txt_col)
    elif cmd == 'txt_clean':
        # txt_clean 10 0 va2021
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        txt_clean_wrapper(ds_name, num_task, job_id)
