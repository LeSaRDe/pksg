import logging
import time
import hashlib

import pandas as pd

from util import global_settings
from util.multitasking import multitasking


"""
PKSG PIPELINE
STAGE: TEXT HASHING (TXT_HASH)

REQUIREMENTS:
    TXT_CLEAN

OUTPUTS:
    hashed text key -> cleaned text and txt ids table: g_txt_hash_file_fmt
        pandas DataFrame
        Index: hashed text key (int): g_hash_txt_col
        Columns: cleaned text (str): g_clean_txt_col
                 text ids corresponding to this text (list): g_txt_idx
                 
    
"""


def txt_hash_single_task(task_id):
    logging.debug('[hash_txt_single_task] Task %s: Starts...' % str(task_id))
    timer_start = time.time()

    df_hash_txt_task = pd.read_pickle(global_settings.g_txt_hash_task_file_fmt.format(task_id))
    logging.debug('[hash_txt_single_task] Task %s: Load in %s tasks.' % (task_id, len(df_hash_txt_task)))

    l_hash_txt_rec = []
    for txt_id, clean_txt_rec in df_hash_txt_task.iterrows():
        clean_txt = clean_txt_rec[global_settings.g_clean_txt_col]
        if clean_txt is None:
            continue
        hash_txt = int(hashlib.md5(clean_txt.encode("utf-8")).hexdigest(), 16)
        l_hash_txt_rec.append((txt_id, hash_txt, clean_txt))

    df_hash_txt = pd.DataFrame(l_hash_txt_rec, columns=[global_settings.g_txt_idx,
                                                        global_settings.g_hash_txt_col,
                                                        global_settings.g_clean_txt_col])
    pd.to_pickle(df_hash_txt, global_settings.g_txt_hash_int_file_fmt.format(task_id))
    logging.debug('[hash_txt_single_task] Task %s: All done in %s secs.' % (task_id, time.time() - timer_start))


def txt_hash_after_merge(df_merge, output_ds_name):
    logging.debug('[txt_hash_after_merge] Starts.')
    timer_start = time.time()

    l_hash_txt = df_merge[global_settings.g_hash_txt_col].to_list()
    d_hash_txt = {hash_txt: [None, []] for hash_txt in l_hash_txt}
    for txt_id, hash_txt_rec in df_merge.iterrows():
        hash_txt = hash_txt_rec[global_settings.g_hash_txt_col]
        clean_txt = hash_txt_rec[global_settings.g_clean_txt_col]
        d_hash_txt[hash_txt][0] = clean_txt
        d_hash_txt[hash_txt][1].append(txt_id)

    df_hash_txt = pd.DataFrame.from_dict(d_hash_txt, orient='index',
                                         columns=[global_settings.g_clean_txt_col,
                                                  global_settings.g_txt_idx])
    df_hash_txt.index.set_names(global_settings.g_hash_txt_col, inplace=True)
    pd.to_pickle(df_hash_txt, global_settings.g_txt_hash_file_fmt.format(output_ds_name))
    logging.debug('[txt_hash_after_merge] All done with %s hash texts in %s secs.'
                  % (len(df_hash_txt), time.time() - timer_start))
    df_merge.drop(global_settings.g_clean_txt_col, 1, inplace=True)
    return df_merge


def txt_hash_wrapper(ds_name, num_task, job_id):
    logging.debug('[txt_hash_wrapper] Starts')
    timer_start = time.time()

    multitasking(single_task_func=txt_hash_single_task,
                 single_task_params=None,
                 num_task=num_task,
                 l_task_ids=None,
                 job_id=job_id,
                 task_type='proc',
                 int_folder=global_settings.g_txt_hash_int_folder,
                 int_fmt=global_settings.g_txt_hash_int_fmt_regex,
                 after_merge_func=txt_hash_after_merge,
                 after_merge_func_params=(ds_name,),
                 out_path=global_settings.g_txt_id_to_hash_txt_file_fmt.format(ds_name),
                 index_col=global_settings.g_txt_idx,
                 rm_int=False)

    logging.debug('[txt_hash_wrapper] All done in %s secs.' % str(time.time() - timer_start))
