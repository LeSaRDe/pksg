import logging
import sys
import math
import pandas as pd

from util import global_settings
from core.txt_hash import txt_hash_wrapper


def gen_txt_hash_tasks(ds_name, num_task, job_id):
    logging.debug('[gen_txt_hash_tasks] Starts.')

    df_clean_txt = pd.read_pickle(global_settings.g_txt_clean_file_fmt.format(ds_name))
    num_txt = len(df_clean_txt)
    logging.debug('[gen_txt_hash_tasks] Load in %s tasks.' % num_txt)

    num_task = int(num_task)
    batch_size = math.ceil(num_txt / num_task)
    l_tasks = []
    for i in range(0, num_txt, batch_size):
        if i + batch_size < num_txt:
            l_tasks.append(df_clean_txt.iloc[i:i + batch_size])
        else:
            l_tasks.append(df_clean_txt.iloc[i:])

    for idx, task in enumerate(l_tasks):
        task_id = str(job_id) + '#' + str(idx)
        pd.to_pickle(task, global_settings.g_txt_hash_task_file_fmt.format(task_id))
    logging.debug('[gen_txt_hash_tasks] %s txt hash tasks are ready.' % len(l_tasks))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        # gen_tasks 10 0 va2021
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        gen_txt_hash_tasks(ds_name, num_task, job_id)
    elif cmd == 'txt_hash':
        # txt_hash 10 0 va2021
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        txt_hash_wrapper(ds_name, num_task, job_id)
