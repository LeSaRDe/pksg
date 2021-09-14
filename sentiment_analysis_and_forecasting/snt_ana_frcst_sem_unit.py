import logging
import time
import math
import sys
import pandas as pd

import scenario_settings
from util import global_settings
from core.semantic_units_extraction import sem_unit_ext_wrapper


def gen_sem_unit_extraction_tasks(ds_name, num_tasks, job_id):
    logging.debug('[gen_sem_unit_extraction_tasks] Starts.')
    timer_start = time.time()

    df_hash_txt = pd.read_pickle(global_settings.g_txt_hash_file_fmt.format(ds_name))
    num_hash_txt = len(df_hash_txt)
    logging.debug('[gen_sem_unit_extraction_tasks] Load in %s clean texts.' % str(num_hash_txt))

    num_tasks = int(num_tasks)
    batch_size = math.ceil(num_hash_txt / num_tasks)
    l_tasks = []
    for i in range(0, num_hash_txt, batch_size):
        if i + batch_size < num_hash_txt:
            l_tasks.append(df_hash_txt.iloc[i:i + batch_size])
        else:
            l_tasks.append(df_hash_txt.iloc[i:])
    logging.debug('[gen_sem_unit_extraction_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for idx, df_task in enumerate(l_tasks):
        pd.to_pickle(df_task, global_settings.g_sem_unit_task_file_fmt.format(str(job_id) + '#' + str(idx)))
    logging.debug('[gen_sem_unit_extraction_tasks] All done with %s tasks in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        # gen_tasks 10 0 va2021
        num_tasks = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        gen_sem_unit_extraction_tasks(ds_name, num_tasks, job_id)
    elif cmd == 'sem_unit':
        # sem_unit 10 0 va2021 covid
        num_tasks = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        if len(sys.argv) >= 6:
            cust_ph_ds_name = sys.argv[5]
        else:
            cust_ph_ds_name = None
        sem_unit_ext_wrapper(ds_name, num_tasks, job_id, scenario_settings.g_cust_ph_file_fmt.format(cust_ph_ds_name))
