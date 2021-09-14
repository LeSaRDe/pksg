import logging
import time

import networkx as nx
import pandas as pd

from util import global_settings
from core.phrase_sentiment import compute_sentiment_for_one_phrase_in_one_txt, sentiment_calibration
from util.multitasking import multitasking

"""
PKSG PIPELINE
STAGE: PKSG (PKSG)

REQUIREMENTS:
    TXT_PH, PH_SENT, SGRAPH

OUTPUTS:
    text->pksg table: g_pksg_file_fmt
        pandas DataFrame
        Index: hash text (int): g_hash_txt_col
        Columns: pksg (NetworkX Graph): g_pksg_col
        
        PKSG
        Node:
          Node label (int): phrase id, e.g. 123.
          Node attributes:
              'pos' (dict): sorted POS keywords linked by single spaces with the number of occurrences as its value,
                          e.g. {'NOUN VERB': 2, 'NOUN NOUN': 1}.
                          Note that the POS tags may not aligned to the lemmas in 'str' in order as both of them have
                          been sorted.
        Edge:
          Edge label (tuple of strings): tuple of phrase ids, e.g. ('ph#123', 'ph#456').
          Edge attributes:
              'sent' (list of of lists of reals): each element is a list of reals representing a sentiment vector of
                          of the format: (very negative, negative, neutral, positive, very positive). An edge may
                          occur multiple times (in one tweet or multiple tweets), which makes this attrinbute a list.
"""


def build_pksg_for_one_txt(task_id, hash_txt, df_phrase, df_txt_id_to_ph_ids, df_ph_sent, df_sgraph, sim_ph_id=None):
    '''
    if 'sim_ph_id' is not specified, then construct a clique upon all phrases of the tweet given by 'tw_id'.
    Otherwise, construct a star graph centered at 'sim_ph_id' which links to every other phrase of the txt.
    '''
    if hash_txt not in df_sgraph.index or hash_txt not in df_ph_sent.index:
        return None

    l_ph_id_pos = [item for item in df_txt_id_to_ph_ids.loc[hash_txt][global_settings.g_ph_id_col]]
    l_ph_ids = [item[0] for item in l_ph_id_pos]
    l_ph_tups = df_phrase.loc[hash_txt][global_settings.g_txt_phrase_col]
    l_sgraph = df_sgraph.loc[hash_txt][global_settings.g_sgraph_col]
    l_parsed_sgraph_info = []
    for sgraph_info in l_sgraph:
        sgraph = sgraph_info[0]
        sgraph_start = sgraph_info[1]
        sgraph_end = sgraph_info[2]
        l_parsed_sgraph_info.append((sgraph, sgraph_start, sgraph_end))
    l_ph_sent = df_ph_sent.loc[hash_txt][global_settings.g_ph_sent_col]

    def edge_sentiment_aggregation(ph_pstn_in_txt_i, ph_pstn_in_txt_j):
        ph_sent_i = l_ph_sent[ph_pstn_in_txt_i][3]
        ph_sent_j = l_ph_sent[ph_pstn_in_txt_j][3]
        if ph_sent_i is None or ph_sent_j is None:
            return None
        _, edge_sent = sentiment_calibration(ph_sent_i, ph_sent_j)
        return edge_sent

    def add_ph_j_into_pksg(ph_pstn_in_txt_j, ph_pstn_in_txt_i, phid_i, phstart_i, phend_i, pksg):
        ph_tup_j = l_ph_tups[ph_pstn_in_txt_j]
        phid_j, phpos_j = l_ph_id_pos[ph_pstn_in_txt_j]
        if phid_i == phid_j:
            return pksg
        phspan_j = ph_tup_j[2]
        phstart_j = min([item[0] for item in phspan_j])
        phend_j = max([item[1] for item in phspan_j])

        edge_phstart = min(phstart_i, phstart_j)
        edge_phend = max(phend_i, phend_j)
        edge_phspan = [(phstart_i, phend_i), (phstart_j, phend_j)]
        edge_phsent = compute_sentiment_for_one_phrase_in_one_txt(l_parsed_sgraph_info, edge_phstart,
                                                                  edge_phend, edge_phspan)
        if edge_phsent is None:
            edge_phsent = edge_sentiment_aggregation(ph_pstn_in_txt_i, ph_pstn_in_txt_j)
        if not pksg.has_edge(phid_i, phid_j):
            if edge_phsent is not None:
                pksg.add_edge(phid_i, phid_j, sent=[edge_phsent], weight=1)
            else:
                pksg.add_edge(phid_i, phid_j, sent=[], weight=1)
        else:
            if edge_phsent is not None:
                pksg.edges()[phid_i, phid_j]['sent'].append(edge_phsent)
                # TODO
                # The weight should be multiplied by the number of texts that refer to this text (e.g. retweets).
                pksg.edges()[phid_i, phid_j]['weight'] += 1
        return pksg

    pksg = nx.Graph()
    # since a phrase id may occur multiple times in a text, we add nodes and edges separately.

    # add nodes
    for ph_pstn_in_txt in range(len(l_ph_tups)):
        phid, phpos = l_ph_id_pos[ph_pstn_in_txt]
        # phstr_i = df_ph_id_to_ph_str.loc[phid][global_settings.g_ph_str_col]
        if not pksg.has_node(phid):
            # pksg.add_node(phid, str=phstr_i, pos={phpos: 1})
            pksg.add_node(phid, pos={phpos: 1})
        else:
            if phpos not in pksg.nodes(data=True)[phid]['pos']:
                pksg.nodes(data=True)[phid]['pos'][phpos] = 1
            else:
                pksg.nodes(data=True)[phid]['pos'][phpos] += 1

    # add edges
    if sim_ph_id is None:
        for ph_pstn_in_txt_i in range(len(l_ph_tups) - 1):
            ph_tup_i = l_ph_tups[ph_pstn_in_txt_i]
            phid_i, phpos_i = l_ph_id_pos[ph_pstn_in_txt_i]
            phspan_i = ph_tup_i[2]
            phstart_i = min([item[0] for item in phspan_i])
            phend_i = max([item[1] for item in phspan_i])

            for ph_pstn_in_txt_j in range(ph_pstn_in_txt_i + 1, len(l_ph_tups)):
                pksg = add_ph_j_into_pksg(ph_pstn_in_txt_j, ph_pstn_in_txt_i, phid_i, phstart_i,
                                          phend_i, pksg)
    else:
        if sim_ph_id not in l_ph_ids:
            raise Exception('[build_pksg_for_one_txt] Task %s: %s is not in %s.' % (task_id, sim_ph_id, hash_txt))
        sim_ph_pstn_in_txt = l_ph_ids.index(sim_ph_id)
        sim_phtup = l_ph_tups[sim_ph_pstn_in_txt]
        _, sim_phpos = l_ph_id_pos[sim_ph_pstn_in_txt]
        sim_phspan = sim_phtup[2]
        sim_phstart = min([item[0] for item in sim_phspan])
        sim_phend = max([item[1] for item in sim_phspan])

        for ph_pstn_in_txt_j in range(len(l_ph_tups)):
            pksg = add_ph_j_into_pksg(ph_pstn_in_txt_j, sim_ph_pstn_in_txt, sim_ph_id, sim_phstart,
                                      sim_phend, pksg)
    return pksg


def pksg_single_task(task_id, sim_ph_id=None, df_phrase=None, df_txt_id_to_ph_ids=None,
                     df_ph_sent=None, df_sgraph=None):
    logging.debug('[pksg_single_task] Task %s: starts.' % str(task_id))
    timer_start = time.time()

    l_hash_txt = []
    with open(global_settings.g_pksg_task_file_fmt.format(str(task_id)), 'r') as in_fd:
        for ln in in_fd:
            l_hash_txt.append(int(ln.strip()))
        in_fd.close()
    logging.debug('[pksg_single_task] Task %s: load in %s txt ids for tasks.' % (task_id, len(l_hash_txt)))

    if df_phrase is None or df_txt_id_to_ph_ids is None or df_ph_sent is None \
            or df_sgraph is None:
        raise Exception('[pksg_single_task] Task %s: Required resources are not ready!' % str(task_id))

    l_pksg_rec = []
    for hash_txt in l_hash_txt:
        pksg = build_pksg_for_one_txt(task_id, hash_txt, df_phrase, df_txt_id_to_ph_ids, df_ph_sent, df_sgraph, sim_ph_id)
        if pksg is None:
            continue
        l_pksg_rec.append((hash_txt, pksg))
        if len(l_pksg_rec) % 500 == 0 and len(l_pksg_rec) >= 500:
            logging.debug('[pksg_single_task] Task %s: %s pksgs done in %s secs.'
                          % (task_id, len(l_pksg_rec), time.time() - timer_start))
    logging.debug('[pksg_single_task] Task %s: %s pksgs done in %s secs.'
                  % (task_id, len(l_pksg_rec), time.time() - timer_start))

    df_pksg = pd.DataFrame(l_pksg_rec, columns=[global_settings.g_hash_txt_col, global_settings.g_pksg_col])
    df_pksg.to_pickle(global_settings.g_pksg_int_file_fmt.format(task_id))
    logging.debug('[pksg_single_task] Task %s: all done with %s pksgs in %s secs.'
                  % (task_id, len(df_pksg), time.time() - timer_start))


def load_resources(job_id, txt_ph_ds_name, ph_sent_ds_name, sgraph_ds_name):
    timer_start = time.time()

    df_phrase = pd.read_pickle(global_settings.g_txt_phrase_file_fmt.format(txt_ph_ds_name))
    logging.debug('[load_resources] Job %s: load in df_phrase with %s recs in %s secs.'
                  % (job_id, len(df_phrase), time.time() - timer_start))
    df_hash_txt_to_ph_ids = pd.read_pickle(global_settings.g_hash_txt_to_ph_ids_file_fmt.format(txt_ph_ds_name))
    logging.debug('[load_resources] Job %s: load in df_hash_txt_to_ph_ids with %s recs in %s secs.'
                  % (job_id, len(df_hash_txt_to_ph_ids), time.time() - timer_start))
    df_ph_sent = pd.read_pickle(global_settings.g_phrase_sent_file_fmt.format(ph_sent_ds_name))
    logging.debug('[load_resources] Job %s: load in df_ph_sent with %s recs in %s secs.'
                  % (job_id, len(df_ph_sent), time.time() - timer_start))
    df_sgraph = pd.read_pickle(global_settings.g_sgraph_file_fmt.format(sgraph_ds_name))
    logging.debug('[load_resources] Job %s: load in df_sgraph with %s recs in %s secs.'
                  % (job_id, len(df_sgraph), time.time() - timer_start))

    return (df_phrase, df_hash_txt_to_ph_ids, df_ph_sent, df_sgraph)


def pksg_wrapper(txt_ph_ds_name, ph_sent_ds_name, sgraph_ds_name, out_ds_name, num_task, job_id,
                 index_col=global_settings.g_hash_txt_col, sim_ph_id=None):
    logging.debug('[pksg_wrapper] Starts')
    timer_start = time.time()

    multitasking(single_task_func=pksg_single_task,
                 single_task_params=(sim_ph_id,),
                 prepare_func=load_resources,
                 prepare_func_params=(job_id, txt_ph_ds_name, ph_sent_ds_name, sgraph_ds_name),
                 num_task=num_task,
                 l_task_ids=None,
                 job_id=job_id,
                 task_type='thread',
                 int_folder=global_settings.g_pksg_int_folder,
                 int_fmt=global_settings.g_pksg_int_fmt_regex.format(str(job_id)),
                 after_merge_func=None,
                 out_path=global_settings.g_pksg_file_fmt.format(out_ds_name),
                 index_col=index_col,
                 rm_int=False)

    logging.debug('[pksg_wrapper] All done in %s secs.' % str(time.time() - timer_start))
