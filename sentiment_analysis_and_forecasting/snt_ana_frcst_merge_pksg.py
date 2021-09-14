import logging
import math
import sys
import time
from os import path

import pandas as pd

from lib import global_settings
from merge_pksg import merge_pksg_wrapper, merge_multiple_pksg
from multitasking import multitasking


def gen_merge_pksg_tasks(ds_name, num_task, job_id, pksg_file_fmt=global_settings.g_pksg_file_fmt):
    logging.debug('[gen_merge_pksg_tasks] starts.')
    timer_start = time.time()

    if not path.exists(pksg_file_fmt.format(ds_name)):
        logging.error('[gen_merge_pksg_tasks] No PKSG for %s! Skip it.' % ds_name)
        return

    df_pksg = pd.read_pickle(pksg_file_fmt.format(ds_name))
    logging.debug('[gen_pksg_tasks] load in df_pksg with %s recs in %s secs.' %
                  (len(df_pksg), time.time() - timer_start))

    l_pksg = df_pksg[global_settings.g_pksg_col].to_list()
    logging.debug('[gen_pksg_tasks] load in %s pksg in %s secs.'
                  % (len(l_pksg), time.time() - timer_start))

    l_pksg = [(item,) for item in l_pksg]
    num_pksg = len(l_pksg)
    num_task = int(num_task)
    batch_size = math.ceil(num_pksg / num_task)
    l_tasks = []
    for i in range(0, num_pksg, batch_size):
        if i + batch_size < num_pksg:
            l_tasks.append(l_pksg[i:i + batch_size])
        else:
            l_tasks.append(l_pksg[i:])
    logging.debug('[gen_pksg_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for task_id, task in enumerate(l_tasks):
        task_name = str(job_id) + '#' + str(task_id)
        df_task = pd.DataFrame(task, columns=[global_settings.g_pksg_col])
        pd.to_pickle(df_task, global_settings.g_merge_pksg_task_file_fmt.format(task_name))
    logging.debug('[gen_pksg_tasks] All done with %s pksg tasks generated.' % str(len(l_tasks)))


# def gen_merge_pksg_ts_tasks(ds_name, num_task):
#     logging.debug('[gen_merge_pksg_ts_tasks] starts.')
#     timer_start = time.time()
#
#     l_pksg_ts_ds_name = []
#     with open(global_settings.g_pksg_ts_ds_name_file_fmt.format(ds_name), 'r') as in_fd:
#         for ln in in_fd:
#             l_pksg_ts_ds_name.append(ln.strip())
#         in_fd.close()
#     logging.debug('[gen_merge_pksg_ts_tasks] Load in %s pksg ts ds_name.' % str(len(l_pksg_ts_ds_name)))
#
#     for pksg_ts_ds_name in l_pksg_ts_ds_name:
#         gen_merge_pksg_tasks(pksg_ts_ds_name, num_task, pksg_ts_ds_name,
#                              pksg_file_fmt=global_settings.g_pksg_ts_file_fmt.format(pksg_ts_ds_name))
#     logging.debug('[gen_merge_pksg_ts_tasks] All done in %s secs.' % str(time.time() - timer_start))
#
#
# def merge_pksg_ts(ds_name, num_task):
#     logging.debug('[merge_pksg_ts] starts.')
#     timer_start = time.time()
#
#     l_pksg_ts_ds_name = []
#     with open(global_settings.g_pksg_ts_ds_name_file_fmt.format(ds_name), 'r') as in_fd:
#         for ln in in_fd:
#             l_pksg_ts_ds_name.append(ln.strip())
#         in_fd.close()
#
#     for pksg_ts_ds_name in l_pksg_ts_ds_name:
#         merge_pksg_int_fmt_regex = r'merge_pksg_int_%s.*\.pickle' % pksg_ts_ds_name
#         merge_pksg_wrapper(pksg_ts_ds_name, num_task, pksg_ts_ds_name, merge_pksg_int_fmt=merge_pksg_int_fmt_regex)
#     logging.debug('[merge_pksg_ts] All done in %s secs.' % str(time.time() - timer_start))


def gen_merge_pksg_ts_tasks(ds_name, pksg_ts_job_cnt, job_id, num_task):
    logging.debug('[gen_merge_pksg_ts_tasks] Start.')

    df_t_int = pd.read_pickle(global_settings.g_t_int_file_fmt.format(ds_name))
    logging.debug('[gen_merge_pksg_ts_tasks] Load in %s time intervals.' % str(len(df_t_int)))


    d_pksg_ts = dict()
    for _, t_int_rec in df_t_int.iterrows():
        t_int_start = t_int_rec['t_int_start']
        t_int_end = t_int_rec['t_int_end']
        t_int_name = t_int_start + '#' + t_int_end
        if t_int_name not in d_pksg_ts:
            d_pksg_ts[t_int_name] = []
        else:
            raise Exception('[gen_merge_pksg_ts_tasks] Duplicate t_int %s' % t_int_name)

    for t_int_name in d_pksg_ts:
        for pksg_ts_job_id in range(int(pksg_ts_job_cnt)):
            pksg_ts_name = t_int_name + '@' + ds_name + '#' + str(pksg_ts_job_id)
            if not path.exists(global_settings.g_pksg_ts_file_fmt.format(pksg_ts_name)):
                logging.error('[gen_merge_pksg_ts_tasks] %s does not exist.'
                              % global_settings.g_pksg_ts_file_fmt.format(pksg_ts_name))
                continue
            df_pskg_ts = pd.read_pickle(global_settings.g_pksg_ts_file_fmt.format(pksg_ts_name))
            l_pksg = df_pskg_ts['pksg'].to_list()
            d_pksg_ts[t_int_name] += l_pksg

    l_pksg_ts_rec = []
    for t_int_name in d_pksg_ts:
        l_pksg_ts_rec.append((t_int_name, d_pksg_ts[t_int_name]))
    df_pksg_ts_merge_tasks = pd.DataFrame(l_pksg_ts_rec, columns=[global_settings.g_t_int_name,
                                                                  global_settings.g_pksg_col])
    df_pksg_ts_merge_tasks = df_pksg_ts_merge_tasks.set_index(global_settings.g_t_int_name)
    logging.debug('[gen_merge_pksg_ts_tasks] Load in all pksg_ts')

    num_t_int = len(d_pksg_ts)
    num_task = int(num_task)
    batch_size = math.ceil(num_t_int / num_task)
    l_tasks = []
    for i in range(0, num_t_int, batch_size):
        if i + batch_size < num_t_int:
            l_tasks.append(df_pksg_ts_merge_tasks.iloc[i:i + batch_size])
        else:
            l_tasks.append(df_pksg_ts_merge_tasks.iloc[i:])
    logging.debug('[gen_merge_pksg_ts_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for task_id, task in enumerate(l_tasks):
        task_name = str(job_id) + '#' + str(task_id)
        pd.to_pickle(task, global_settings.g_merge_pksg_ts_task_file_fmt.format(task_name))
    logging.debug('[gen_merge_pksg_ts_tasks] All done with %s merge pksg_ts tasks generated.' % str(len(l_tasks)))


def merge_pksg_ts_single_task(task_id, batch_size):
    logging.debug('[merge_pksg_ts_single_task] Task %s: Starts.' % task_id)
    timer_start = time.time()

    df_task = pd.read_pickle(global_settings.g_merge_pksg_ts_task_file_fmt.format(task_id))
    logging.debug('[merge_pksg_ts_single_task] Task %s: Load in %s merge pksg_ts tasks.' % (task_id, len(df_task)))

    l_merge_pksg_ts_rec = []
    for t_int_name, pksg_ts_rec in df_task.iterrows():
        l_pksg = pksg_ts_rec[global_settings.g_pksg_col]
        merge_pksg = merge_multiple_pksg(task_id, l_pksg, batch_size)
        if merge_pksg is None:
            logging.error('[merge_pksg_ts_single_task] Task %s: PKSG merge failed on %s.' % (task_id, t_int_name))
        else:
            l_merge_pksg_ts_rec.append((t_int_name, merge_pksg))

    df_merge_pksg_ts_rec = pd.DataFrame(l_merge_pksg_ts_rec, columns=[global_settings.g_t_int_name,
                                                                      global_settings.g_merge_pksg_col])
    pd.to_pickle(df_merge_pksg_ts_rec, global_settings.g_merge_pksg_ts_int_file_fmt.format(task_id))
    logging.debug('[merge_pksg_ts_single_task] Task %s: All done in %s secs.' % (task_id, time.time() - timer_start))


def merge_pksg_ts_wrapper(ds_name, num_task, job_id, batch_size=10):
    logging.debug('[merge_pksg_wrapper] Starts')
    timer_start = time.time()

    multitasking(single_task_func=merge_pksg_ts_single_task,
                 single_task_params=(batch_size,),
                 num_task=num_task,
                 l_task_ids=None,
                 job_id=job_id,
                 task_type='proc',
                 int_folder=global_settings.g_merge_pksg_int_folder,
                 int_fmt=global_settings.g_merge_pksg_ts_int_fmt_regex.format(ds_name),
                 after_merge_func=None,
                 after_merge_func_params=None,
                 out_path=global_settings.g_merge_pksg_ts_file_fmt.format(ds_name),
                 index_col=global_settings.g_t_int_name,
                 rm_int=False)

    logging.debug('[merge_pksg_wrapper] All done in %s secs.' % str(time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        # gen_tasks 10 0 202005
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        gen_merge_pksg_tasks(ds_name, num_task, job_id)
    elif cmd == 'merge_pksg':
        # merge_pksg 10 0 202005
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        merge_pksg_wrapper(ds_name, num_task, job_id)
    elif cmd == 'gen_merge_pksg_ts_tasks':
        # gen_merge_pksg_ts_tasks 10 va2021 0 10
        num_task = sys.argv[2]
        ds_name = sys.argv[3]
        job_id = sys.argv[4]
        pksg_ts_job_cnt = sys.argv[5]
        gen_merge_pksg_ts_tasks(ds_name, pksg_ts_job_cnt, job_id, num_task)
    elif cmd == 'merge_pksg_ts':
        # merge_pksg_ts 10 0 va2021
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        merge_pksg_ts_wrapper(ds_name, num_task, job_id)
