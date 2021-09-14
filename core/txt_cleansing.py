import logging
import time
import pandas as pd

from util import global_settings
from util.multitasking import multitasking
from core.semantic_units_extractor import prelim_txt_clean


"""
PKSG PIPELINE
STAGE: TEXT CLEANSING (TXT_CLEAN)

REQUIREMENTS:
    RAW_TXT

OUTPUTS:
    text->cleaned text table: g_txt_clean_file_fmt
        pandas DataFrame
        Index: text id (str): g_txt_idx
        Columns: cleaned text (str): g_clean_txt_col
"""


def txt_clean_single_task(task_id, raw_txt_col, clean_txt_col_name='clean_txt',
                          extra_clean_func=None, extra_clean_func_params=None):
    '''
    Each task should be a table file of the pickle format containing the target raw text column. This table should
    also be indexed.
    :param task_id: The id used to locate the task file.
    :param raw_txt_col: Specifies the raw text column.
    :param clean_txt_col_name: Specifies the name of column for the cleaned texts.
    :return: A table file with the same index as the input task containing the cleaned texts in the specified column.
    '''
    logging.debug('[txt_clean_single_task] Task %s: Starts...' % str(task_id))
    timer_start = time.time()

    df_txt_clean_task = pd.read_pickle(global_settings.g_txt_clean_task_file_fmt.format(task_id))
    logging.debug('[txt_clean_single_task] Task %s: Load in %s tasks.' % (task_id, len(df_txt_clean_task)))

    if raw_txt_col is None or raw_txt_col not in df_txt_clean_task:
        raise Exception('[txt_clean_single_task] %s is not a valid column of df_tw_clean_task' % raw_txt_col)

    l_txt_clean_rec = []
    # df_tw_clean_task[txt_col] is a Series.
    for txt_idx, raw_txt in df_txt_clean_task[raw_txt_col].iteritems():
        clean_txt = prelim_txt_clean(raw_txt, extra_clean_func, extra_clean_func_params)
        if clean_txt is not None and clean_txt != '':
            l_txt_clean_rec.append((txt_idx, clean_txt))
        else:
            l_txt_clean_rec.append((txt_idx, None))
    if clean_txt_col_name is None or clean_txt_col_name == '':
        logging.error('[txt_clean_single_task] clean_txt_col_name is None or an empty string! %s is used.'
                      % global_settings.g_clean_txt_col)
        clean_txt_col_name = global_settings.g_clean_txt_col
    df_clean_txt = pd.DataFrame(l_txt_clean_rec, columns=[df_txt_clean_task.index.name, clean_txt_col_name])
    # df_clean_txt = df_clean_txt.set_index(df_tw_clean_task.index.name)
    pd.to_pickle(df_clean_txt, global_settings.g_txt_clean_int_file_fmt.format(task_id))
    logging.debug('[txt_clean_single_task] Task %s: All done with %s recs in %s secs.'
                  % (task_id, len(df_clean_txt), time.time() - timer_start))


def txt_clean_wrapper(ds_name, num_task, job_id, index_col=global_settings.g_txt_idx,
                      raw_txt_col=global_settings.g_raw_txt_col,
                      clean_txt_col_name=global_settings.g_clean_txt_col,
                      extra_clean_func=None,
                      after_merge_func=None,
                      after_merge_func_params=None):
    logging.debug('[txt_clean_wrapper] Starts')
    timer_start = time.time()

    multitasking(single_task_func=txt_clean_single_task,
                 single_task_params=(raw_txt_col, clean_txt_col_name, extra_clean_func),
                 num_task=num_task,
                 l_task_ids=None,
                 job_id=job_id,
                 task_type='proc',
                 int_folder=global_settings.g_txt_clean_int_folder,
                 int_fmt=global_settings.g_txt_clean_int_fmt_regex,
                 after_merge_func=after_merge_func,
                 after_merge_func_params=after_merge_func_params,
                 out_path=global_settings.g_txt_clean_file_fmt.format(ds_name),
                 index_col=index_col,
                 rm_int=False)

    logging.debug('[txt_clean_wrapper] All done in %s secs.' % str(time.time() - timer_start))

