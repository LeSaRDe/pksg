import logging
import re
import os
from os import walk, path
import multiprocessing
import threading
import pandas as pd


def multitasking(single_task_func, single_task_params, num_task=None, l_task_ids=None, job_id=None, task_type='proc',
                 prepare_func=None, prepare_func_params=None, merge_int=True, int_folder=None, int_fmt=None,
                 after_merge_func=None, after_merge_func_params=None, out_path=None, index_col=None, rm_int=False):
    '''
    Utility function for multitasking. Can be multiprocessing or multithreading. The tasks are identified by an input
    list or a generated list of task ids. The user needs to take care of the inputs of each task in the implementation
    of 'single_task_func'.
    :param single_task_func: The function for a single task.
    :param single_task_params: The additional parameters for the single task function.
    The first param is always task_id.
    :param num_task: The number of tasks to run in parallel. 'None' uses all cores.
    :param l_task_ids: If 'l_task_ids' is specified, 'num_task' is overridden by the length of 'l_task_ids'.
    'l_task_ids' needs to provided each task id to 'single_task_func'. This task id can be important for
    'single_task_func' to locate intermediate results. If 'l_task_ids' is None, then the task ids are generated by
    integers bounded by 'num_task'.
    :param job_id: The user may have multiple jobs, and each runs in a multitasking way. 'job_id' identifies each job
    when necessary. If 'job_id' is not None, each task id is of the format '[job_id]#[task_id]',
    otherwise simply '[task_id]'.
    :param task_type: 'proc' specifies multiprocessing; 'thread' specifies multithreading.
    :param prepare_func: Used for preparation work before running each task. Its return MUST be a tuple if any;
    otherwise the return MUST be None. This tuple will be concatenated to the end of 'single_task_params' for each task.
    For example, necessary resources can be loaded in, and passed to each task, when using 'thread'.
    :param prepare_func_params: The parameters sent into 'prepare_func'.
    :param merge_int: 'True' will merge intermediate results.
    :param int_folder: The folder for intermediate results to merge. If 'int_folder' is None, the merging is skipped.
    :param int_fmt: The file name regex of intermediate results.
    :param after_merge_func: A function can be applied to the merged date before the final output. This function MUST
    return the processed df_merge.
    :param after_merge_func_params: The additional params for after_merge_func. The first param is always df_merge,
    the merged table data.
    :param out_path: The final output path (the full path including the file name) for the merged results. The output
    is a pickle table file in the format of pandas DataFrame.
    :param index_col: The column name for the index in the final output table file.
    :param rm_int: 'True' will remove all intermediate results after merging.
    :return: No direct reture value. All results will be stored into files.
    '''
    if single_task_func is None:
        raise Exception('[multitasking] single_task_func is missing!')
    if num_task is None and l_task_ids is None:
        raise Exception('[multitasking] num_task or l_task_ids needs to be specified!')
    if l_task_ids is None:
        if int(num_task) <= 0:
            raise Exception('[multitasking] num_task needs to be a positive integer!')
    else:
        if type(l_task_ids) != list:
            raise Exception('[multitasking] l_task_ids needs to be a list!')
        elif len(l_task_ids) <= 0:
            raise Exception('[multitasking] l_task_ids should not be empty!')

    if task_type == 'proc':
        task_carrier = multiprocessing.Process
    elif task_type == 'thread':
        task_carrier = threading.Thread
    else:
        raise Exception('[multitasking] task_type can only be "proc" or "thread"!')

    if prepare_func is not None:
        if prepare_func_params is None:
            prepare_func_params = ()
        prepare_ret = prepare_func(*prepare_func_params)
    else:
        prepare_ret = None

    if single_task_params is None:
        single_task_params = ()

    if l_task_ids is None:
        if isinstance(job_id, str) or isinstance(job_id, int) or job_id is None:
            l_task_ids = [str(job_id) + '#' + str(idx) if job_id is not None else str(idx) for idx in
                          range(int(num_task))]
        elif isinstance(job_id, list):
            l_task_ids = []
            for each_job_id in job_id:
                l_task_ids += [str(each_job_id) + '#' + str(idx) for idx in range(int(num_task))]

    l_task_instance = []
    for task_id in l_task_ids:
        if prepare_ret is not None:
            single_task_actual_params = (task_id,) + single_task_params + prepare_ret
        else:
            single_task_actual_params = (task_id,) + single_task_params
        task_instance = task_carrier(target=single_task_func,
                                     args=single_task_actual_params,
                                     name='Task ' + str(task_id))
        task_instance.start()
        single_task_actual_params = ()
        l_task_instance.append(task_instance)

    while len(l_task_instance) > 0:
        for task_instance in l_task_instance:
            if task_instance.is_alive():
                task_instance.join(1)
            else:
                l_task_instance.remove(task_instance)

    if merge_int:
        df_merge = merge_int_rets(int_folder, int_fmt, index_col, rm_int)
    else:
        df_merge = None

    if after_merge_func is not None:
        if after_merge_func_params is not None:
            after_merge_func_params = (df_merge,) + after_merge_func_params
        else:
            after_merge_func_params = (df_merge,)
        df_merge = after_merge_func(*after_merge_func_params)
    if df_merge is not None and out_path is not None:
        pd.to_pickle(df_merge, out_path)


def merge_int_rets(int_folder, int_fmt, index_col, rm_int):
    if int_folder is not None and int_fmt is not None:
        if not path.exists(int_folder):
            logging.error('[merge_int_rets] int_folder does not exist! Skip the merging!')
            return None
        l_int = []
        for (dirpath, dirname, filenames) in walk(int_folder):
            for filename in filenames:
                if re.match(int_fmt, filename) is None:
                    continue
                df_int = pd.read_pickle(dirpath + filename)
                l_int.append(df_int)
                if rm_int:
                    os.remove(dirpath + filename)
        if len(l_int) <= 0:
            logging.error('[merge_int_rets] No intermediate result to merge.')
            return None
        df_merge = pd.concat(l_int)
        if index_col is not None:
            if index_col not in df_merge:
                logging.error('[merge_int_rets] index_col is not a valid column name! Skip setting index!')
            else:
                df_merge = df_merge.set_index(index_col)

        return df_merge
    else:
        return None
