import json
import logging
import time

import networkx as nx
import pandas as pd

import global_settings
from multitasking import multitasking


"""
PKSG PIPELINE
STAGE: SENTIMENT GRAPH (SGRAPH)

REQUIREMENTS:
    TXT_SENT

OUTPUTS:
    text->sentiment graph table: g_sgraph_file_fmt
        pandas DataFrame
        Index: hash text (int): g_hash_txt_col
        Columns: sgraph tuples (list of tuples): g_sgraph_col
        
        Each sgraph tuple contains three fields. The first field is a NetworkX DiGraph which is the actual sentiment
        graph of a sentence in the text. The second and third fields are the start and end character positions. 
"""


def build_sgraph_from_json_str(sgraph_json_str):
    '''
    Return a directed graph with the start_char_idx and the end_char_idx relative to the text as a whole.
    '''
    # logging.debug('[build_sgraph_from_json] Starts...')

    sgraph_json = json.loads(sgraph_json_str)
    l_snodes = sgraph_json['nodes']
    l_sedges = sgraph_json['edges']

    sgraph = nx.DiGraph()
    for snode in l_snodes:
        snode_id = snode['id']
        snode_pos = snode['pos']
        snode_sentiments = snode['sentiments']
        snode_token_str = snode['token_str']
        snode_token_start = snode['token_start']
        snode_token_end = snode['token_end']
        sgraph.add_node(snode_id, pos=snode_pos, sentiments=snode_sentiments, token_str=snode_token_str,
                        token_start=snode_token_start, token_end=snode_token_end)

    for sedge in l_sedges:
        src_id = sedge['src_id']
        trg_id = sedge['trg_id']
        sgraph.add_edge(src_id, trg_id)

    # build a segment tree over the binary constituent sentiment tree
    if len(sgraph.nodes()) == 1 and len(sgraph.edges()) == 0:
        for node in sgraph.nodes(data=True):
            if node[1]['token_start'] != -1 and node[1]['token_end'] != -1:
                return sgraph, node[1]['token_start'], node[1]['token_end']

    d_parents = dict()
    for node in sgraph.nodes(data=True):
        if node[1]['token_start'] != -1 and node[1]['token_end'] != -1:
            l_parents = list(sgraph.predecessors(node[0]))
            if len(l_parents) != 1:
                raise Exception('[build_sgraph_from_json] Leaf has multiple parents: %s, sgraph_json_str: %s'
                                % (node, sgraph_json_str))
            parent = l_parents[0]
            if parent not in d_parents:
                d_parents[parent] = [node]
            else:
                d_parents[parent].append(node)

    while len(d_parents) > 0:
        s_done_parents = set([])
        d_new_parents = dict()
        for parent in d_parents:
            if len(list(sgraph.successors(parent))) == len(d_parents[parent]):
                parent_start = min([child[1]['token_start'] for child in d_parents[parent]])
                parent_end = max([child[1]['token_end'] for child in d_parents[parent]])
                sgraph.nodes(data=True)[parent]['token_start'] = parent_start
                sgraph.nodes(data=True)[parent]['token_end'] = parent_end
                s_done_parents.add(parent)
                if sgraph.nodes(data=True)[parent]['pos'] == 'ROOT':
                    continue
                l_new_parents = list(sgraph.predecessors(parent))
                if len(l_new_parents) != 1:
                    raise Exception('[build_sgraph_from_json] Leaf has multiple parents: %s' % str(parent))
                new_parent = l_new_parents[0]
                if new_parent in d_new_parents:
                    d_new_parents[new_parent].append((parent, sgraph.nodes(data=True)[parent]))
                else:
                    d_new_parents[new_parent] = [(parent, sgraph.nodes(data=True)[parent])]
        for parent in s_done_parents:
            d_parents.pop(parent)
        for new_parent in d_new_parents:
            if new_parent not in d_parents:
                d_parents[new_parent] = d_new_parents[new_parent]
            else:
                d_parents[new_parent] += d_new_parents[new_parent]

    graph_start = min([node[1]['token_start'] for node in sgraph.nodes(data=True)])
    graph_end = max([node[1]['token_end'] for node in sgraph.nodes(data=True)])
    # logging.debug('[build_sgraph_from_json] sgraph is done: %s' % nx.info(sgraph))
    return sgraph, graph_start, graph_end


def build_sgraph_from_json_single_task(task_id, hash_txt_col, sgraph_col):
    logging.debug('[build_sgraph_from_json_single_task] Task %s: Starts.' % str(task_id))
    timer_start = time.time()

    with open(global_settings.g_txt_sent_int_file_fmt.format(task_id), 'r') as in_fd:
        l_txt_sent = json.load(in_fd)
        in_fd.close()
    logging.debug('[build_sgraph_from_json_single_task] Task %s: Load in %s txt sent recs.'
                  % (task_id, len(l_txt_sent)))

    ready_cnt = 0
    l_ready = []
    for d_txt_sent_rec in l_txt_sent:
        hash_txt = int(d_txt_sent_rec[hash_txt_col])
        l_sgraph_json_str = d_txt_sent_rec[sgraph_col]
        if len(l_sgraph_json_str) <= 0:
            logging.error('[build_sgraph_from_json_single_task] Task %s: No sgraph for text: %s' % (task_id, hash_txt))
            continue
        l_sgraph = []
        for sgraph_json_str in l_sgraph_json_str:
            sgraph, graph_start, graph_end = build_sgraph_from_json_str(sgraph_json_str)
            # sgraph_out_str = json.dumps(nx.adjacency_data(sgraph))
            l_sgraph.append((sgraph, graph_start, graph_end))
        l_ready.append((hash_txt, l_sgraph))
        ready_cnt += 1
        if ready_cnt % 100 == 0 and ready_cnt >= 100:
            logging.debug('[build_sgraph_from_json_single_task] Task %s: %s sgraphs done in %s secs.'
                          % (task_id, ready_cnt, time.time() - timer_start))
    logging.debug('[build_sgraph_from_json_single_task] Task %s: %s sgraphs done in %s secs.'
                  % (task_id, ready_cnt, time.time() - timer_start))
    df_sgraph = pd.DataFrame(l_ready, columns=[hash_txt_col, sgraph_col])
    pd.to_pickle(df_sgraph, global_settings.g_sgraph_int_file_fmt.format(task_id))
    logging.debug('[build_sgraph_from_json_single_task] Task %s: All done with %s sgraphs in %s secs.'
                  % (task_id, len(df_sgraph), time.time() - timer_start))


def build_sgraph_from_json_wrapper(ds_name, num_task, job_id, index_col=global_settings.g_hash_txt_col,
                                   sgraph_col=global_settings.g_sgraph_col):
    logging.debug('[build_sgraph_from_json_wrapper] Starts')
    timer_start = time.time()

    multitasking(single_task_func=build_sgraph_from_json_single_task,
                 single_task_params=(index_col, sgraph_col),
                 num_task=num_task,
                 l_task_ids=None,
                 job_id=job_id,
                 task_type='proc',
                 int_folder=global_settings.g_txt_sent_int_folder,
                 int_fmt=global_settings.g_sgraph_int_fmt_regex.format(str(job_id)),
                 after_merge_func=None,
                 out_path=global_settings.g_sgraph_file_fmt.format(ds_name),
                 index_col=index_col,
                 rm_int=False)

    logging.debug('[build_sgraph_from_json_wrapper] All done in %s secs.' % str(time.time() - timer_start))
