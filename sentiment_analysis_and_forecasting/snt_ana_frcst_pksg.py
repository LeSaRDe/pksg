import logging
import math
import sys
import time

import pandas as pd

from lib import global_settings
from pksg import pksg_wrapper


def gen_pksg_tasks(txt_ph_ds_name, ph_sent_ds_name, sgraph_ds_name, num_tasks, job_id):
    logging.debug('[gen_pksg_tasks] Starts.')
    timer_start = time.time()

    df_phrase = pd.read_pickle(global_settings.g_txt_phrase_file_fmt.format(txt_ph_ds_name))
    logging.debug('[gen_pksg_tasks] load in df_phrase with %s recs in %s secs.' %
                  (len(df_phrase), time.time() - timer_start))

    df_ph_sent = pd.read_pickle(global_settings.g_phrase_sent_file_fmt.format(ph_sent_ds_name))
    logging.debug('[gen_pksg_tasks] load in df_ph_sent with %s recs in %s secs.'
                  % (len(df_ph_sent), time.time() - timer_start))
    df_sgraph = pd.read_pickle(global_settings.g_sgraph_file_fmt.format(sgraph_ds_name))
    logging.debug('[gen_pksg_tasks] load in df_sgraph with %s recs in %s secs.'
                  % (len(df_sgraph), time.time() - timer_start))

    l_hash_txt = list(set(df_phrase.index.to_list()).intersection(df_ph_sent.index.to_list())
                      .intersection(df_sgraph.index.to_list()))
    num_txt_id = len(l_hash_txt)
    num_tasks = int(num_tasks)
    batch_size = math.ceil(num_txt_id / num_tasks)
    l_tasks = []
    for i in range(0, num_txt_id, batch_size):
        if i + batch_size < num_txt_id:
            l_tasks.append(l_hash_txt[i:i + batch_size])
        else:
            l_tasks.append(l_hash_txt[i:])
    logging.debug('[gen_pksg_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for task_id, task in enumerate(l_tasks):
        task_name = str(job_id) + '#' + str(task_id)
        with open(global_settings.g_pksg_task_file_fmt.format(task_name), 'w+') as out_fd:
            out_str = '\n'.join([str(item) for item in task])
            out_fd.write(out_str)
            out_fd.close()
    logging.debug('[gen_pksg_tasks] All done with %s tw pksg tasks generated.' % str(len(l_tasks)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        # gen_tasks 1 0 puretest
        num_tasks = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        txt_ph_ds_name = ds_name
        ph_sent_ds_name = ds_name + '#' + job_id
        sgraph_ds_name = ds_name + '#' + job_id
        gen_pksg_tasks(txt_ph_ds_name, ph_sent_ds_name, sgraph_ds_name, num_tasks, job_id)
    elif cmd == 'pksg':
        # pksg 1 0 puretest
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        txt_ph_ds_name = ds_name
        ph_sent_ds_name = ds_name + '#' + job_id
        sgraph_ds_name = ds_name + '#' + job_id
        out_ds_name = ds_name + '#' + job_id
        pksg_wrapper(txt_ph_ds_name, ph_sent_ds_name, sgraph_ds_name, out_ds_name, num_task, job_id)
