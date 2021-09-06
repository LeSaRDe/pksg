import logging
import sys
import time
import math

import pandas as pd

import scenario_settings
import global_settings
from phrase_sentiment import phrase_sent_wrapper


def gen_phrase_sent_tasks(sgraph_ds_name, txt_ph_ds_name, num_tasks, job_id):
    logging.debug('[gen_phrase_sent_tasks] Starts.')
    timer_start = time.time()

    df_sgraph = pd.read_pickle(global_settings.g_sgraph_file_fmt.format(sgraph_ds_name))
    num_sgraph = len(df_sgraph)
    logging.debug('[gen_phrase_sent_tasks] Load in %s sgraph recs.' % str(num_sgraph))

    df_phrase = pd.read_pickle(global_settings.g_txt_phrase_file_fmt.format(txt_ph_ds_name))
    num_phrase = len(df_phrase)
    logging.debug('[gen_phrase_sent_tasks] Load in %s phrase recs.' % str(num_phrase))

    l_task_ready = []
    for hash_txt, phrase_rec in df_phrase.iterrows():
        if hash_txt not in df_sgraph.index:
            continue
        phrase = phrase_rec[global_settings.g_txt_phrase_col]
        sgarph = df_sgraph.loc[hash_txt][global_settings.g_sgraph_col]
        l_task_ready.append((hash_txt, phrase, sgarph))
    num_recs = len(l_task_ready)
    logging.debug('[gen_phrase_sent_tasks] %s texts for tasks.' % str(num_recs))

    num_tasks = int(num_tasks)
    batch_size = math.ceil(num_recs / num_tasks)
    l_tasks = []
    for i in range(0, num_recs, batch_size):
        if i + batch_size < num_recs:
            l_tasks.append(l_task_ready[i:i + batch_size])
        else:
            l_tasks.append(l_task_ready[i:])
    logging.debug('[gen_phrase_sent_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for idx, task in enumerate(l_tasks):
        df_task = pd.DataFrame(task, columns=[global_settings.g_hash_txt_col, global_settings.g_txt_phrase_col,
                                              global_settings.g_sgraph_col])
        df_task = df_task.set_index(global_settings.g_hash_txt_col)
        pd.to_pickle(df_task, global_settings.g_phrase_sent_task_file_fmt.format(str(job_id) + '#' + str(idx)))
    logging.debug('[gen_phrase_sent_tasks] All done with %s tasks in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        # gen_tasks 1 0 puretest
        num_tasks = sys.argv[2]
        job_id = sys.argv[3]
        df_name = sys.argv[4]
        sgraph_ds_name = df_name + '#' + job_id
        txt_ph_ds_name = df_name
        gen_phrase_sent_tasks(sgraph_ds_name, txt_ph_ds_name, num_tasks, job_id)
    elif cmd == 'ph_sent':
        # ph_sent 1 0 puretest
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4] + '#' + job_id
        phrase_sent_wrapper(ds_name, num_task, job_id)
