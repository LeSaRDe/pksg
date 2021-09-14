import logging
import time
import pandas as pd
from os import path

from util import global_settings
from util.multitasking import multitasking
from core.semantic_units_extractor import SemUnitsExtractor


"""
PKSG PIPELINE
STAGE: SEMANTIC UNITS EXTRACTION (SEM_UNIT)

REQUIREMENTS:
    TXT_HASH

OUTPUTS:
    text->semantic units table: g_sem_unit_file_fmt
        pandas DataFrame
        Index: text hash key (int): g_hash_txt_col
        Columns: core clause graph (NetworkX Graph): g_sem_unit_cls_col
                 noun phrase (list of noun phrase tuples): g_sem_unit_nps_col
        
        Each noun phrase tuple is: (lemma string linked by white space, set of word indices, phrase start position,
        phrase end position)
"""


def sem_unit_ext_single_task(task_id, clean_txt_col, l_cust_ph):
    logging.debug('[sem_unit_ext_single_task] Task %s: Starts.' % str(task_id))
    timer_start = time.time()

    df_hash_txt = pd.read_pickle(global_settings.g_sem_unit_task_file_fmt.format(task_id))
    logging.debug('[sem_unit_ext_single_task] Task %s: Load in %s clean texts.'
                  % (task_id, len(df_hash_txt)))
    l_texts = []
    for hash_txt_key, hash_txt_rec in df_hash_txt.iterrows():
        clean_txt = hash_txt_rec[clean_txt_col]
        l_texts.append((hash_txt_key, clean_txt))

    sem_unit_ext_ins = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)
    df_sem_units = sem_unit_ext_ins.sem_unit_extraction_batch(l_texts, task_id, df_hash_txt.index.name, l_cust_ph)
    pd.to_pickle(df_sem_units, global_settings.g_sem_unit_int_file_fmt.format(task_id))
    logging.debug('[sem_unit_ext_single_task] Task %s: All done in %s secs.'
                  % (task_id, time.time() - timer_start))


def sem_unit_ext_wrapper(ds_name, num_task, job_id, cust_ph_path=None, index_col=global_settings.g_hash_txt_col,
                         clean_txt_col=global_settings.g_clean_txt_col):
    logging.debug('[sem_unit_ext_wrapper] Starts')
    timer_start = time.time()

    l_cust_ph = []
    if cust_ph_path is not None and path.exists(cust_ph_path):
        with open(cust_ph_path, 'r') as in_fd:
            for ln in in_fd:
                l_cust_ph.append(ln.strip())
            in_fd.close()

    multitasking(single_task_func=sem_unit_ext_single_task,
                 single_task_params=(clean_txt_col, l_cust_ph),
                 num_task=num_task,
                 l_task_ids=None,
                 job_id=job_id,
                 task_type='proc',
                 int_folder=global_settings.g_sem_unit_int_folder,
                 int_fmt=global_settings.g_sem_unit_int_fmt_regex,
                 after_merge_func=None,
                 out_path=global_settings.g_sem_unit_file_fmt.format(ds_name),
                 index_col=index_col,
                 rm_int=False)

    logging.debug('[sem_unit_ext_wrapper] All done in %s secs.' % str(time.time() - timer_start))