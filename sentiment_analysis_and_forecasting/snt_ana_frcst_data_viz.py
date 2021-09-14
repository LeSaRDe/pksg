import logging
import time
import sys
from os import path

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from util import global_settings

"""
PKSG PIPELINE
STAGE: DATA VISUALIZATION (DATA_VIZ)

REQUIREMENTS:
    FILTER_PKSG, MERGE_PKSG, T_INT
"""


def draw_pksg_with_polarized_edges(ds_name, pksg, pksg_ds_name):
    logging.debug('[draw_pksg_with_polarized_edges] Starts.')
    timer_start = time.time()

    df_ph_id_to_ph_str = pd.read_pickle(global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ds_name))
    logging.debug('[draw_pksg_with_polarized_edges] Load in df_ph_id_to_ph_str with %s recs.'
                  % str(len(df_ph_id_to_ph_str)))

    def color_edge_by_sentiment(l_sent_vec):
        if l_sent_vec is None or len(l_sent_vec) <= 0:
            return None
        mean_sent_vec = np.mean(l_sent_vec, axis=0)
        sent_class = np.argmax(np.asarray(mean_sent_vec))
        if sent_class == 0:
            return 'darkblue'
        elif sent_class == 1:
            return 'cornflowerblue'
        elif sent_class == 2:
            return 'darkgrey'
        elif sent_class == 3:
            return 'lightcoral'
        elif sent_class == 4:
            return 'darkred'
        else:
            return None

    l_edge = []
    l_edge_color = []
    l_node = []
    for edge in pksg.edges():
        edge_color = color_edge_by_sentiment(pksg.edges[edge]['sent'])
        if edge_color is not None:
            l_edge.append(edge)
            l_node.append(edge[0])
            l_node.append(edge[1])
            l_edge_color.append(edge_color)
    l_node = list(set(l_node))
    d_node_label = {ph_id: df_ph_id_to_ph_str.loc[ph_id][global_settings.g_ph_str_col] for ph_id in l_node}

    sub_pksg = nx.subgraph(pksg, l_node)
    logging.debug('[draw_pksg_with_polarized_edges] Draw sub_pksg: %s.' % nx.info(sub_pksg))

    plt.figure(1, figsize=(50, 50), tight_layout={'pad': 1, 'w_pad': 50, 'h_pad': 50, 'rect': None})
    graph_layout = nx.spring_layout(sub_pksg, k=0.6)
    # graph_layout = graphviz_layout(sub_pksg, prog="sfdp")
    logging.debug('[draw_pksg_with_polarized_edges] graph_layout done.')

    nx.draw_networkx_edges(sub_pksg, pos=graph_layout, edgelist=l_edge, edge_color=l_edge_color, width=2.0)
    logging.debug('[draw_pksg_with_polarized_edges] draw_networkx_edges done.')

    nx.draw_networkx_nodes(sub_pksg, pos=graph_layout, nodelist=l_node, node_size=10)
    logging.debug('[draw_pksg_with_polarized_edges] draw_networkx_nodes done.')

    nx.draw_networkx_labels(sub_pksg, pos=graph_layout, labels=d_node_label, font_size=20)
    logging.debug('[draw_pksg_with_polarized_edges] draw_networkx_labels done.')

    plt.savefig(global_settings.g_data_viz_folder + 'polarized_pksg_{0}.png'.format(pksg_ds_name), format="PNG")
    logging.debug('[draw_pksg_with_polarized_edges] All done for %s in %s secs'
                  % (pksg_ds_name, time.time() - timer_start))
    plt.clf()
    plt.close()


def draw_filtered_pksg_ts_with_polarized_edges(ds_name):
    logging.debug('[draw_pksg_ts_with_polarized_edges] Starts.')
    timer_start = time.time()

    l_pksg_ds_name = []
    with open(global_settings.g_pksg_ts_ds_name_file_fmt.format(ds_name), 'r') as in_fd:
        for ln in in_fd:
            l_pksg_ds_name.append(ln.strip())
        in_fd.close()
    logging.debug('[draw_pksg_ts_with_polarized_edges] Load in %s pskg ts ds_name.' % str(len(l_pksg_ds_name)))

    # l_pksg_ds_name = ['20200508050118#20200508050618@202005']
    for pksg_ds_name in l_pksg_ds_name:
        if not path.exists(global_settings.g_filtered_pksg_file_fmt.format(pksg_ds_name)):
            logging.error('[draw_pksg_ts_with_polarized_edges] No filtered PKSG for %s' % pksg_ds_name)
            continue
        pksg = nx.read_gpickle(global_settings.g_filtered_pksg_file_fmt.format(pksg_ds_name))
        logging.debug('[draw_pksg_with_polarized_edges] Load in pksg for %s: %s.' % (pksg_ds_name, nx.info(pksg)))
        draw_pksg_with_polarized_edges(ds_name, pksg, pksg_ds_name)
    logging.debug('[draw_pksg_ts_with_polarized_edges] All done in %s secs.' % str(time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    plt.set_loglevel('error')
    cmd = sys.argv[1]

    if cmd == 'filtered_pksg_ts_pol_edges':
        # filtered_pksg_ts_pol_edges 202005
        ds_name = sys.argv[2]
        draw_filtered_pksg_ts_with_polarized_edges(ds_name)
