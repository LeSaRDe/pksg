import logging
import math
import re
import time
from os import walk, path

import networkx as nx
import pandas as pd

import global_settings
from multitasking import multitasking


"""
PKSG PIPELINE
STAGE: MERGE PKSG (MERGE_PKSG)

REQUIREMENTS:
    PKSG, T_INT

OUTPUTS:
    A pksg over all considered texts: g_merge_pksg_file_fmt
        NetworkX Graph gpickle
"""


def merge_two_pksg(pksg_1, pksg_2):
    if pksg_1.number_of_nodes() == 0 or pksg_2.number_of_nodes() == 0:
        raise Exception('[merge_two_tw_pksg] empty pksg occurs.')

    for node_1 in pksg_1.nodes(data=True):
        if not pksg_2.has_node(node_1):
            pksg_2.add_node(node_1[0], pos=node_1[1]['pos'])
        else:
            for pos_1 in node_1[1]['pos']:
                if pos_1 not in pksg_2.nodes(data=True)[node_1[0]]:
                    pksg_2.nodes(data=True)[node_1[0]][pos_1] = node_1[1]['pos'][pos_1]
                else:
                    pksg_2.nodes(data=True)[node_1[0]][pos_1] += node_1[1]['pos'][pos_1]

    for edge_1 in pksg_1.edges(data=True):
        if not pksg_2.has_edge(edge_1[0], edge_1[1]):
            pksg_2.add_edge(edge_1[0], edge_1[1], sent=edge_1[2]['sent'], weight=edge_1[2]['weight'])
        else:
            pksg_2.edges()[edge_1[0], edge_1[1]]['sent'] += edge_1[2]['sent']
            pksg_2.edges()[edge_1[0], edge_1[1]]['weight'] += edge_1[2]['weight']

    return pksg_2


def divide_and_conquer_merge_pksg(l_pksg):
    if len(l_pksg) < 1:
        raise Exception('[divide_and_conquer_merge_pksg] Invalid l_pksg!')
    if len(l_pksg) == 1:
        return l_pksg[0]
    if len(l_pksg) == 2:
        return merge_two_pksg(l_pksg[0], l_pksg[1])

    batch_size = math.ceil(len(l_pksg) / 2)
    ret_graph = merge_two_pksg(divide_and_conquer_merge_pksg(l_pksg[:batch_size]),
                               divide_and_conquer_merge_pksg(l_pksg[batch_size:]))
    return ret_graph


def merge_multiple_pksg(task_id, l_pksg, batch_size=10):
    if l_pksg is None or len(l_pksg) <= 0:
        logging.error('[merge_multiple_pksg] Task %s: No pksg to merge.' % task_id)
        return None
    timer_start = time.time()
    cnt = 0
    while True:
        # As divide_and_conquer_merge_pksg is implemented by a recursion, to prevent the stack overflow,
        # the input task is further partitioned into sub-tasks.
        l_tasks = []
        num_tasks = len(l_pksg)
        for i in range(0, num_tasks, batch_size):
            if i + batch_size < num_tasks:
                l_tasks.append(l_pksg[i:i + batch_size])
            else:
                l_tasks.append(l_pksg[i:])

        l_rets = []
        for task in l_tasks:
            merge_pksg = divide_and_conquer_merge_pksg(task)
            l_rets.append(merge_pksg)

        if len(l_rets) <= 0:
            raise Exception('[merge_multiple_pksg] Task %s: Invalid l_rets' % task_id)
        elif len(l_rets) == 1:
            return l_rets[0]
        else:
            cnt += 1
            logging.debug(
                '[merge_multiple_pksg] Task %s: %s iterations done in %s secs. %s pksg left to merge.'
                % (task_id, cnt, time.time() - timer_start, len(l_rets)))
            l_pksg = l_rets


def merge_pksg_single_task(task_id, batch_size=10):
    logging.debug('[merge_pksg_single_task] Task %s: starts.' % str(task_id))
    timer_start = time.time()

    if not path.exists(global_settings.g_merge_pksg_task_file_fmt.format(task_id)):
        logging.error('[merge_pksg_single_task] Task %s: No task found. Skip it.' % str(task_id))
        return
    df_merge_pksg_task = pd.read_pickle(global_settings.g_merge_pksg_task_file_fmt.format(task_id))
    logging.debug('[merge_pksg_single_task] Task %s: load in %s pksg in %s secs.'
                  % (task_id, len(df_merge_pksg_task), time.time() - timer_start))

    l_pksg = df_merge_pksg_task[global_settings.g_pksg_col].to_list()
    if len(l_pksg) <= 0:
        logging.error('[merge_pksg_single_task] Task %s: No PKSG to merge. Skip it.' % str(task_id))
        return

    merge_pksg = merge_multiple_pksg(task_id, l_pksg, batch_size)
    if merge_pksg is None:
        logging.error('[merge_pksg_single_task] Task %s: PKSG merge failed.' % str(task_id))
        return

    nx.write_gpickle(merge_pksg, global_settings.g_merge_pksg_int_file_fmt.format(task_id))
    logging.debug('[merge_pksg_single_task] Task %s: All done in %s secs: %s'
                  % (task_id, time.time() - timer_start, nx.info(merge_pksg)))

    # cnt = 0
    # while True:
    #     # As divide_and_conquer_merge_pksg is implemented by a recursion, to prevent the stack overflow,
    #     # the input task is further partitioned into sub-tasks.
    #     l_tasks = []
    #     num_tasks = len(l_pksg)
    #     for i in range(0, num_tasks, batch_size):
    #         if i + batch_size < num_tasks:
    #             l_tasks.append(l_pksg[i:i + batch_size])
    #         else:
    #             l_tasks.append(l_pksg[i:])
    #
    #     l_rets = []
    #     for task in l_tasks:
    #         merge_pksg = divide_and_conquer_merge_pksg(task)
    #         l_rets.append(merge_pksg)
    #
    #     if len(l_rets) <= 0:
    #         raise Exception('[merge_pksg_single_task] Task %s: invalid l_rets')
    #     elif len(l_rets) == 1:
    #         nx.write_gpickle(l_rets[0], global_settings.g_merge_pksg_int_file_fmt.format(task_id))
    #         logging.debug('[merge_pksg_single_task] Task %s: All done in %s secs: %s'
    #                       % (task_id, time.time() - timer_start, nx.info(l_rets[0])))
    #         return
    #     else:
    #         cnt += 1
    #         logging.debug(
    #             '[merge_pksg_single_task] Task %s: %s iterations done in %s secs. %s pksg left to merge.'
    #             % (task_id, cnt, time.time() - timer_start, len(l_rets)))
    #         l_pksg = l_rets


def final_merge_pksg(df_merge, ds_name, merge_pksg_int_fmt):
    l_merge_pksg_int = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_merge_pksg_int_folder):
        for filename in filenames:
            if re.match(merge_pksg_int_fmt, filename) is None:
                continue
            merge_pksg_int = nx.read_gpickle(dirpath + filename)
            l_merge_pksg_int.append(merge_pksg_int)
    if len(l_merge_pksg_int) <= 0:
        logging.error('[final_merge_pksg] No intermediate PKSG for the final merge. Need to check!')
        return df_merge
    merge_pksg = divide_and_conquer_merge_pksg(l_merge_pksg_int)
    nx.write_gpickle(merge_pksg, global_settings.g_merge_pksg_file_fmt.format(ds_name))
    return df_merge


def merge_pksg_wrapper(ds_name, num_task, job_id, batch_size=10,
                       merge_pksg_int_fmt=global_settings.g_merge_pksg_int_fmt_regex):
    logging.debug('[merge_pksg_wrapper] Starts')
    timer_start = time.time()

    multitasking(single_task_func=merge_pksg_single_task,
                 single_task_params=(batch_size,),
                 num_task=num_task,
                 l_task_ids=None,
                 job_id=job_id,
                 task_type='proc',
                 int_folder=None,
                 int_fmt=None,
                 after_merge_func=final_merge_pksg,
                 after_merge_func_params=(ds_name, merge_pksg_int_fmt),
                 out_path=None,
                 index_col=None,
                 rm_int=False)

    logging.debug('[merge_pksg_wrapper] All done in %s secs.' % str(time.time() - timer_start))
