import logging
import time
import math
import sys

import pandas as pd

from util import global_settings
from core.phrase_extraction import phrase_ext_wrapper


def gen_phrase_ext_tasks(ds_name, num_tasks, job_id):
    logging.debug('[gen_phrase_ext_tasks] Starts.')
    timer_start = time.time()

    df_sem_unit = pd.read_pickle(global_settings.g_sem_unit_file_fmt.format(ds_name))
    num_sem_unit = len(df_sem_unit)
    logging.debug('[gen_phrase_ext_tasks] Load in %s recs.' % str(num_sem_unit))

    num_tasks = int(num_tasks)
    batch_size = math.ceil(num_sem_unit / num_tasks)
    l_tasks = []
    for i in range(0, num_sem_unit, batch_size):
        if i + batch_size < num_sem_unit:
            l_tasks.append(df_sem_unit.iloc[i:i + batch_size])
        else:
            l_tasks.append(df_sem_unit.iloc[i:])
    logging.debug('[gen_phrase_ext_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for idx, df_task in enumerate(l_tasks):
        pd.to_pickle(df_task, global_settings.g_txt_phrase_task_file_fmt.format(str(job_id) + '#' + str(idx)))
    logging.debug('[gen_phrase_ext_tasks] All done with %s tasks in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        # gen_tasks 1 0 puretest
        num_tasks = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        gen_phrase_ext_tasks(ds_name, num_tasks, job_id)
    elif cmd == 'phrase_ext':
        # phrase_ext 1 0 puretest
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        phrase_ext_wrapper(ds_name, num_task, job_id)
