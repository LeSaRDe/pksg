import logging
import math
import sys
import time
from os import path

import networkx as nx
import pandas as pd

import scenario_settings
import global_settings
from filter_pksg import filter_pksg_node_wrapper, filter_pksg_edge_wrapper, PKSGNodeFilterBase, PKSGEdgeFilterBase
from semantic_units_extractor import SemUnitsExtractor


"""
PKSG PIPELINE
STAGE: FILTER PKSG (FILTER_PKSG)

REQUIREMENTS:
    MERGE_PKSG, T_INT, TXT_PH

OUTPUTS:
    A filtered PKSG for each time interval w.r.t. a given query (i.e. a string). 
"""


"""""""""""""""""""""
PKSG NODE FILTERING
"""""""""""""""""""""
def gen_filter_pksg_node_tasks(ds_name, num_task, pksg_ds_name):
    logging.debug('[gen_filter_pksg_node_tasks] Starts for PKSG %s' % pksg_ds_name)

    pksg = nx.read_gpickle(global_settings.g_merge_pksg_file_fmt.format(pksg_ds_name))
    logging.debug('[gen_filter_pksg_node_tasks] Load in PKSG: %s' % nx.info(pksg))

    df_ph_id_to_ph_str = pd.read_pickle(global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ds_name))
    logging.debug('[gen_filter_pksg_node_tasks] Load in df_ph_id_to_ph_str with %s recs' % str(len(df_ph_id_to_ph_str)))

    l_node_obj = list(pksg.nodes(data=True))
    l_node_obj = [(node_obj[0], df_ph_id_to_ph_str.loc[node_obj[0]]['ph_str'], node_obj[1]) for node_obj in l_node_obj]

    num_node = len(l_node_obj)
    num_task = int(num_task)
    batch_size = math.ceil(num_node / num_task)
    l_tasks = []
    for i in range(0, num_node, batch_size):
        if i + batch_size < num_node:
            l_tasks.append(l_node_obj[i:i + batch_size])
        else:
            l_tasks.append(l_node_obj[i:])
    logging.debug('[gen_filter_pksg_node_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for task_id, task in enumerate(l_tasks):
        task_name = str(pksg_ds_name) + '#' + str(task_id)
        df_task = pd.DataFrame(task, columns=[global_settings.g_node_id_col, 'node_str', 'node_attr'])
        df_task = df_task.set_index(global_settings.g_node_id_col)
        pd.to_pickle(df_task, global_settings.g_filter_node_task_file_fmt.format(task_name))
    logging.debug('[gen_filter_pksg_node_tasks] All done with %s filter PKSG node tasks generated.' % str(len(df_task)))


def gen_filter_pksg_ts_node_tasks(ds_name, num_task):
    logging.debug('[gen_filter_pksg_ts_node_tasks] Starts.')

    l_pksg_ds_name = []
    with open(global_settings.g_pksg_ts_ds_name_file_fmt.format(ds_name), 'r') as in_fd:
        for ln in in_fd:
            l_pksg_ds_name.append(ln.strip())
        in_fd.close()

    for pksg_ds_name in l_pksg_ds_name:
        if not path.exists(global_settings.g_merge_pksg_file_fmt.format(pksg_ds_name)):
            logging.debug('[gen_filter_pksg_ts_node_tasks] No PKSG for %s. Skip it.' % pksg_ds_name)
            continue
        gen_filter_pksg_node_tasks(ds_name, num_task, pksg_ds_name)
    logging.debug('[gen_filter_pksg_ts_node_tasks] All done.')


class NodeFilterByQuery(PKSGNodeFilterBase):
    """
    TODO
    Can be customized.
    Need to implement filterings for kept POS, removed POS and POS min count.
    """
    def __init__(self, query_str=None, l_kept_pos=None, l_rm_pos=None, d_pos_min_cnt=None):
        super().__init__()
        if query_str is not None:
            sem_unit_ext_ins = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)
            l_query_lemma = sem_unit_ext_ins.extract_sal_lemmas_from_sent(query_str)
            self.m_s_query_lemma = set(l_query_lemma)
        else:
            self.m_s_query_lemma = None
        if l_kept_pos is not None:
            self.l_kept_pos = l_kept_pos
        else:
            self.l_kept_pos = None
        if l_rm_pos is not None:
            self.l_rm_pos = l_rm_pos
        else:
            self.l_rm_pos = None
        if d_pos_min_cnt is not None:
            self.d_pos_min_cnt = d_pos_min_cnt
        else:
            self.d_pos_min_cnt = None

    def node_filter(self, node_obj):
        node_str = node_obj['node_str']
        s_node_token = set([token.strip() for token in node_str.split(' ')])
        if len(s_node_token.intersection(self.m_s_query_lemma)) > 0:
            return True
        else:
            return False


def filter_pksg_node(ds_name, num_task, query):
    logging.debug('[filter_pksg_node] Starts.')
    timer_start = time.time()

    sem_unit_ext_ins = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)
    l_query_lemma = sem_unit_ext_ins.extract_sal_lemmas_from_sent(query)
    query = ' '.join(l_query_lemma)

    l_pksg_ds_name = []
    with open(global_settings.g_pksg_ts_ds_name_file_fmt.format(ds_name), 'r') as in_fd:
        for ln in in_fd:
            l_pksg_ds_name.append(ln.strip())
        in_fd.close()

    for pksg_ds_name in l_pksg_ds_name:
        filter_pksg_node_wrapper(pksg_ds_name, num_task, pksg_ds_name, NodeFilterByQuery, (query,))
        logging.debug('[filter_pksg_node] Done filtered nodes for %s in %s secs.'
                      % (pksg_ds_name, time.time() - timer_start))
    logging.debug('[filter_pksg_node] All done in %s secs.' % str(time.time() - timer_start))


"""""""""""""""""""""
PKSG EDGE FILTERING
"""""""""""""""""""""
def gen_filter_pksg_edge_tasks(num_task, pksg_ds_name):
    logging.debug('[gen_filter_pksg_edge_tasks] Starts for PKSG %s' % pksg_ds_name)

    pksg = nx.read_gpickle(global_settings.g_merge_pksg_file_fmt.format(pksg_ds_name))
    logging.debug('[gen_filter_pksg_edge_tasks] Load in PKSG: %s' % nx.info(pksg))

    df_filtered_nodes = pd.read_pickle(global_settings.g_filtered_nodes_file_fmt.format(pksg_ds_name))
    s_filtered_nodes = set(df_filtered_nodes[global_settings.g_ph_id_col].to_list())
    logging.debug('[gen_filter_pksg_edge_tasks] Load in %s filtered nodes.' % str(len(s_filtered_nodes)))

    l_task_edges = []
    for edge in pksg.edges(data=True):
        node_1_id = edge[0]
        node_2_id = edge[1]
        if node_1_id not in s_filtered_nodes or node_2_id not in s_filtered_nodes:
            continue
        l_task_edges.append(((node_1_id, node_2_id), edge[2]))
    if len(l_task_edges) <= 0:
        logging.debug('[gen_filter_pksg_edge_tasks] No task edge for %s. Skip it.' % pksg_ds_name)
        return

    num_edge = len(l_task_edges)
    num_task = int(num_task)
    batch_size = math.ceil(num_edge / num_task)
    l_tasks = []
    for i in range(0, num_edge, batch_size):
        if i + batch_size < num_edge:
            l_tasks.append(l_task_edges[i:i + batch_size])
        else:
            l_tasks.append(l_task_edges[i:])
    logging.debug('[gen_filter_pksg_edge_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for task_id, task in enumerate(l_tasks):
        task_name = str(pksg_ds_name) + '#' + str(task_id)
        df_task = pd.DataFrame(task, columns=[global_settings.g_edge_id_col, 'edge_attr'])
        df_task = df_task.set_index(global_settings.g_edge_id_col)
        pd.to_pickle(df_task, global_settings.g_filter_edge_task_file_fmt.format(task_name))
    logging.debug('[gen_filter_pksg_edge_tasks] All done with %s filter PKSG edge tasks generated.' % str(len(df_task)))


def gen_filter_pksg_ts_edge_tasks(ds_name, num_task):
    logging.debug('[gen_filter_pksg_ts_edge_tasks] Starts.')

    l_pksg_ds_name = []
    with open(global_settings.g_pksg_ts_ds_name_file_fmt.format(ds_name), 'r') as in_fd:
        for ln in in_fd:
            l_pksg_ds_name.append(ln.strip())
        in_fd.close()

    for pksg_ds_name in l_pksg_ds_name:
        if not path.exists(global_settings.g_merge_pksg_file_fmt.format(pksg_ds_name)):
            logging.debug('[gen_filter_pksg_ts_edge_tasks] No PKSG for %s. Skip it.' % pksg_ds_name)
            continue
        gen_filter_pksg_edge_tasks(num_task, pksg_ds_name)
    logging.debug('[gen_filter_pksg_ts_edge_tasks] All done.')


class EdgeFilter(PKSGEdgeFilterBase):
    """
    TODO
    Can be customized.
    Need to implement this filter appropriately.
    """
    def __init__(self):
        super().__init__()

    def edge_filter(self, edge_attr):
        return True


def filter_pksg_edge(ds_name, num_task, EdgeFilterParams):
    """
    TODO
    Need to implement how to take EdgeFilterParams.
    """
    logging.debug('[filter_pksg_edge] Starts.')
    timer_start = time.time()

    l_pksg_ds_name = []
    with open(global_settings.g_pksg_ts_ds_name_file_fmt.format(ds_name), 'r') as in_fd:
        for ln in in_fd:
            l_pksg_ds_name.append(ln.strip())
        in_fd.close()

    for pksg_ds_name in l_pksg_ds_name:
        filter_pksg_edge_wrapper(pksg_ds_name, num_task, pksg_ds_name, EdgeFilter, EdgeFilterParams)
        logging.debug('[filter_pksg_edge] Done filtered edges for %s in %s secs.'
                      % (pksg_ds_name, time.time() - timer_start))
    logging.debug('[filter_pksg_edge] All done in %s secs.' % str(time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_filter_node_tasks':
        # gen_filter_node_tasks 10 202005
        num_task = sys.argv[2]
        ds_name = sys.argv[3]
        gen_filter_pksg_ts_node_tasks(ds_name, num_task)
    elif cmd == 'filter_pksg_node':
        # filter_pksg_node 10 202005 "wearing masks"
        num_task = sys.argv[2]
        ds_name = sys.argv[3]
        query = sys.argv[4]
        filter_pksg_node(ds_name, num_task, query)
    elif cmd == 'gen_filter_edge_tasks':
        # gen_filter_edge_tasks 10 202005
        num_task = sys.argv[2]
        ds_name = sys.argv[3]
        gen_filter_pksg_ts_edge_tasks(ds_name, num_task)
