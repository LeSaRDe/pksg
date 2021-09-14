import logging
import math
import time

import networkx as nx
import pandas as pd

from util import global_settings
from util.multitasking import multitasking


"""
PKSG PIPELINE
STAGE: FILTER PKSG (FILTER_PKSG)

REQUIREMENTS:
    T_INT

OUTPUTS:
    A filtered PKSG for each time interval w.r.t. a given query. The nodes consist of two parts.
    One is the set of nodes (called center nodes) that matches the query (i.e. sharing at least one lemma). 
    The other consists of the neighbors of each center node. 
"""


def divide_and_conquer_filter(df_obj, filter_ins, iter_id=0):
    """
    Return the list of object ids that matches the query.
    """
    # TODO
    # To prevent stack overflow, the depth of recursion is fixed. This number of depth deserves a more careful thinking.
    if iter_id >= 20 or len(df_obj) < 2:
        l_kept_obj_id = []
        for obj_id, obj_rec in df_obj.iterrows():
            if df_obj.node_filter(obj_rec):
                l_kept_obj_id.append(obj_id)
        return l_kept_obj_id

    half_len = math.ceil(len(df_obj) / 2)
    iter_id += 1
    l_kept_obj_id_1st_half = divide_and_conquer_filter(df_obj.iloc[:half_len], filter_ins, iter_id)
    l_kept_obj_id_2nd_half = divide_and_conquer_filter(df_obj.iloc[half_len:], filter_ins, iter_id)
    return l_kept_obj_id_1st_half + l_kept_obj_id_2nd_half



"""""""""""""""""""""
PKSG NODE FILTERING
"""""""""""""""""""""
class PKSGNodeFilterBase:
    def node_filter(self, node_obj):
        """
        :param node_obj: (tuple) (node_str, node_attribute_dict)
        :return: True -- keep the node; False -- remove the node.
        """
        return True


def filter_pksg_node_single_task(task_id, node_filter_class, node_filter_params):
    logging.debug('[filter_pksg_node_single_task] Task %s: starts.' % str(task_id))
    timer_start = time.time()

    df_node_obj = pd.read_pickle(global_settings.g_filter_node_task_file_fmt.format(task_id))
    logging.debug('[filter_pksg_node_single_task] Task %s: Load in %s node objects.' % (task_id, len(df_node_obj)))

    if node_filter_params is None:
        node_filter_params = ()
    node_filter_ins = node_filter_class(*node_filter_params)
    l_kept_node = divide_and_conquer_filter(df_node_obj, node_filter_ins)
    l_kept_node = [(node_id,) for node_id in l_kept_node]
    df_kept_node = pd.DataFrame(l_kept_node, columns=[global_settings.g_node_id_col])
    pd.to_pickle(df_kept_node, global_settings.g_filter_node_int_file_fmt.format(task_id))
    logging.error('[filter_pksg_node_single_task] Task %s: All done with %s kept nodes in %s sec.'
                  % (task_id, len(df_kept_node), time.time() - timer_start))


def filter_pksg_node_wrapper(pksg_ds_name, num_task, job_id, node_filter_class, node_filter_params):
    logging.debug('[filter_pksg_node_wrapper] Starts')
    timer_start = time.time()

    multitasking(single_task_func=filter_pksg_node_single_task,
                 single_task_params=(node_filter_class, node_filter_params),
                 num_task=num_task,
                 l_task_ids=None,
                 job_id=job_id,
                 task_type='proc',
                 int_folder=global_settings.g_filter_pksg_int_folder,
                 int_fmt=global_settings.g_filter_node_int_fmt_regex.format(pksg_ds_name),
                 after_merge_func=None,
                 after_merge_func_params=None,
                 out_path=global_settings.g_filtered_nodes_file_fmt.format(pksg_ds_name),
                 index_col=None,
                 rm_int=False)

    logging.debug('[filter_pksg_node_wrapper] All done in %s secs.' % str(time.time() - timer_start))


"""""""""""""""""""""
PKSG EDGE FILTERING
"""""""""""""""""""""
class PKSGEdgeFilterBase:
    def edge_filter(self, edge_attr):
        """
        :param edge_attr: (dict) NetworkX Edge attribute dict.
        :return: True -- keep the edge; False -- remove the edge.
        """
        return True


def filter_pksg_edge_single_task(task_id, edge_filter_class, edge_filter_params):
    logging.debug('[filter_pksg_edge_single_task] Task %s: starts.' % str(task_id))
    timer_start = time.time()

    df_edge_obj = pd.read_pickle(global_settings.g_filter_edge_task_file_fmt.format(task_id))
    logging.debug('[filter_pksg_edge_single_task] Task %s: Load in %s edge objects.' % (task_id, len(df_edge_obj)))

    if edge_filter_params is None:
        edge_filter_params = ()
    edge_filter_ins = edge_filter_class(*edge_filter_params)
    l_kept_edge = divide_and_conquer_filter(df_edge_obj, edge_filter_ins)
    l_kept_edge = [(edge_id,) for edge_id in l_kept_edge]
    df_kept_edge = pd.DataFrame(l_kept_edge, columns=[global_settings.g_edge_id_col])
    pd.to_pickle(df_kept_edge, global_settings.g_filter_edge_int_file_fmt.format(task_id))
    logging.error('[filter_pksg_edge_single_task] Task %s: All done with %s kept edges in %s sec.'
                  % (task_id, len(df_kept_edge), time.time() - timer_start))


def filter_pksg_edge_wrapper(pksg_ds_name, num_task, job_id, edge_filter_class, edge_filter_params):
    logging.debug('[filter_pksg_edge_wrapper] Starts')
    timer_start = time.time()

    multitasking(single_task_func=filter_pksg_edge_single_task,
                 single_task_params=(edge_filter_class, edge_filter_params),
                 num_task=num_task,
                 l_task_ids=None,
                 job_id=job_id,
                 task_type='proc',
                 int_folder=global_settings.g_filter_pksg_int_folder,
                 int_fmt=global_settings.g_filter_edge_int_fmt.format(pksg_ds_name),
                 after_merge_func=None,
                 after_merge_func_params=None,
                 out_path=global_settings.g_filtered_edges_file_fmt.format(pksg_ds_name),
                 index_col=None,
                 rm_int=False)

    logging.debug('[filter_pksg_edge_wrapper] All done in %s secs.' % str(time.time() - timer_start))


def sub_pksg_on_filtered_nodes(df_filtered_nodes, pksg_ds_name):
    pksg = nx.read_gpickle(global_settings.g_merge_pksg_file_fmt.format(pksg_ds_name))
    l_center_nodes = df_filtered_nodes.index.to_list()
    l_filtered_nodes = []
    l_filtered_nodes += l_center_nodes
    for center_node in l_center_nodes:
        neighbors = nx.neighbors(pksg, center_node)
        l_filtered_nodes += list(neighbors)
        l_filtered_nodes = list(set(l_filtered_nodes))
    sub_pksg = pksg.subgraph(l_filtered_nodes)
    nx.write_gpickle(sub_pksg, global_settings.g_filtered_pksg_file_fmt.format(pksg_ds_name))
    return df_filtered_nodes