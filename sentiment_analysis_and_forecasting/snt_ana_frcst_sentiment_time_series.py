import logging
import math
import sys
import time
from os import path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scenario_settings
import global_settings
from multitasking import multitasking


"""
PKSG PIPELINE
STAGE: SENTIMENT TIME SERIES (SENT_TS)

REQUIREMENTS:
    T_INT

OUTPUTS:
    A time interval table: g_t_int_file_fmt
        pandas DataFrame
        Index: time interval id (int): g_t_int_id
        Columns: 
            time interval start (str): g_t_int_start_col, format: YYYYMMDDHHMMSS
            time interval end (str): g_t_int_end_col, format: YYYYMMDDHHMMSS
            text ids within the time interval: (list of strs): g_t_int_txt_ids
    
    A PKSG table for each time interval: g_pksg_ts_file_fmt
        (same as g_pksg_file_fmt)
    
    A txt file storing all time interval PKSGs' names: g_pksg_ts_ds_name_file_fmt
"""


def gen_sent_ts_tasks(pksg_ts_ds_name_prefix, pksg_ts_job_cnt, trg_job_cnt, task_cnt_per_job):
    logging.debug('[gen_sent_ts_tasks] Starts.')
    timer_start = time.time()

    l_pksg_ts_ds_name = []
    for i in range(int(pksg_ts_job_cnt)):
        pksg_ts_ds_name = pksg_ts_ds_name_prefix + '#' + str(i)
        with open(global_settings.g_pksg_ts_ds_name_file_fmt.format(pksg_ts_ds_name), 'r') as in_fd:
            for ln in in_fd:
                l_pksg_ts_ds_name.append(ln.strip())
            in_fd.close()
    total_num_pksg_ts = len(l_pksg_ts_ds_name)
    logging.debug('[gen_sent_ts_tasks] %s pksg_ts ds_name in total.' % str(total_num_pksg_ts))

    trg_job_cnt = int(trg_job_cnt)
    task_cnt_per_job = int(task_cnt_per_job)
    job_batch_size = math.ceil(total_num_pksg_ts / trg_job_cnt)
    if job_batch_size < 1:
        job_batch_size = 1
    l_jobs = []
    for job_id in range(0, total_num_pksg_ts, job_batch_size):
        if job_id + job_batch_size < total_num_pksg_ts:
            l_jobs.append(l_pksg_ts_ds_name[job_id:job_id + job_batch_size])
        else:
            l_jobs.append(l_pksg_ts_ds_name[job_id:])

    for job_id, job in enumerate(l_jobs):
        task_batch_size = math.ceil(len(job) / task_cnt_per_job)

        task_id = 0
        for i in range(0, len(job), task_batch_size):
            if i + task_batch_size < total_num_pksg_ts:
                task = job[i:i + task_batch_size]
            else:
                task = job[i:]
            with open(global_settings.g_sent_ts_task_file_fmt.format(str(job_id) + '#' + str(task_id)), 'w+') as out_fd:
                out_str = '\n'.join(task)
                out_fd.write(out_str)
                out_fd.close()
            task_id += 1
    logging.debug('[gen_sent_ts_tasks] All done in %s secs.' % str(time.time() - timer_start))


def classify_sentiment(sent_vec):
    if sent_vec is None:
        return None

    sent_class = np.argmax(np.asarray(sent_vec))
    if sent_class == 2:
        return None
    elif sent_class == 3:
        return 2
    elif sent_class == 4:
        return 3
    else:
        return sent_class


def sentiments_by_queries_in_pksg(pksg, l_queries, agg_type, output_ds_name):
    """
    For each query phrase, locate it in 'pksg', collect its 1-hop neighbors, and aggregate the sentiments of neighbors.
    :param l_queries: (list of str) The list of query phrase ids.
    :param agg_type: 'pol_cnt' -- For each sentiment vector, classify it to a category (i.e. very neg, neg, pos,
    very pos). Then count for each category. The neutral category is not considered. The 'support' in the return will
    reflect how much the neutral takes the proportion. Finally, normalize the counts to probabilities.
    :return: (pandas DataFrame) Each tuple corresponds to a query phrase.
        Index: 'query' (str)
        Columns:
            'agg_sents': Aggregated sentiments for each query. Its type depends on 'agg_type'.
                - agg_type='pol_cnt': (list of floats)
                  [probability of very neg, probability of neg, probability of pos, probability of very pos]
            'eff_support' (int): The sum of weights of neighbors that contribute sentiments.
            'support' (int): The sum of weights of all neighbors.
    """
    logging.debug('[sentiments_by_queries_in_pksg] Starts for %s.' % output_ds_name)
    timer_start = time.time()
    if pksg is None:
        logging.error('[sentiments_by_queries_in_pksg] Invalid pksg!')
        return None
    if len(pksg.nodes()) <= 0:
        logging.error('[sentiments_by_queries_in_pksg] Empty pksg!')
        return None
    if len(l_queries) <= 0:
        logging.error('[sentiments_by_queries_in_pksg] No query!')
        return None
    l_supported_agg_type = ['pol_cnt']
    if agg_type not in l_supported_agg_type:
        logging.error('[sentiments_by_queries_in_pksg] agg_type %s is not supported!' % agg_type)
        return None

    d_neig_info = dict()
    for query_ph_id in l_queries:
        if not pksg.has_node(query_ph_id):
            continue
        l_sent = []
        support = 0
        for neig in pksg.neighbors(query_ph_id):
            neig_edge = pksg.edges[(query_ph_id, neig)]
            if neig_edge['sent'] is not None and len(neig_edge['sent']) > 0:
                l_sent += neig_edge['sent']
            support += neig_edge['weight']
        d_neig_info[query_ph_id] = (l_sent, support)
    if len(d_neig_info) <= 0:
        logging.debug('[sentiments_by_queries_in_pksg] Did not find sentiment info in PKSGs for any query in %s'
                      % str(l_queries))
        return None

    d_sent_rec = dict()
    if agg_type == 'pol_cnt':
        for query_ph_id in d_neig_info:
            eff_support = 0
            agg_sent_vec = np.zeros(4)
            l_sent = d_neig_info[query_ph_id][0]
            for sent_vec in l_sent:
                sent_class = classify_sentiment(sent_vec)
                if sent_class is not None:
                    eff_support += 1
                    agg_sent_vec[sent_class] += 1
            if query_ph_id not in d_sent_rec:
                d_sent_rec[query_ph_id] = [agg_sent_vec, eff_support, d_neig_info[query_ph_id][1]]
            else:
                d_sent_rec[query_ph_id][0] += agg_sent_vec
                d_sent_rec[query_ph_id][1] += eff_support
                d_sent_rec[query_ph_id][2] += d_neig_info[query_ph_id][1]
    logging.debug('[sentiments_by_queries_in_pksg] All done for %s in %s secs.'
                  % (output_ds_name, time.time() - timer_start))
    return d_sent_rec


def sentiments_by_queries_in_pksg_single_task(task_id, l_queries, query_name, agg_type='pol_cnt'):
    logging.debug('[sentiments_by_queries_in_pksg_single_task] Task %s: Starts.' % task_id)
    timer_start = time.time()

    if not path.exists(global_settings.g_sent_ts_task_file_fmt.format(task_id)):
        logging.error('[sentiments_by_queries_in_pksg_single_task] Task %s: No task' % task_id)
        return

    l_pksg_ts_ds_name = []
    with open(global_settings.g_sent_ts_task_file_fmt.format(task_id), 'r') as in_fd:
        for ln in in_fd:
            l_pksg_ts_ds_name.append(ln.strip())
        in_fd.close()

    for pksg_ts_ds_name in l_pksg_ts_ds_name:
        df_pksg_ts = pd.read_pickle(global_settings.g_pksg_ts_file_fmt.format(pksg_ts_ds_name))
        logging.debug('[sentiments_by_queries_in_pksg_single_task] Task %s: Load in %s pksg_ts.'
                      % (task_id, len(df_pksg_ts)))
        for txt_id, pksg_ts_rec in df_pksg_ts.iterrows():
            pksg_ts = pksg_ts_rec['pksg']
            d_sent_rec = sentiments_by_queries_in_pksg(pksg_ts, l_queries, agg_type, pksg_ts_ds_name)
            if d_sent_rec is None:
                continue
            df_sent_ts = pd.DataFrame.from_dict(d_sent_rec, orient='index',
                                                columns=['agg_sents', 'eff_support', 'support'])
            df_sent_ts.index.set_names('query', inplace=True)
            pd.to_pickle(df_sent_ts, global_settings.g_sent_ts_int_file_fmt.format(pksg_ts_ds_name + '#' + query_name))
    logging.debug('[sentiments_by_queries_in_pksg_single_task] Task %s: All done in %s secs.'
                  % (task_id, time.time() - timer_start))


def sentiments_by_queries_in_pksg_wrapper(num_task, job_id, ph_ds_name, query_name, query_path, agg_type='pol_cnt'):
    logging.debug('[sentiments_by_queries_in_pksg_wrapper] Starts.')
    timer_start = time.time()

    if query_path is None or not path.exists(query_path):
        raise Exception('[sentiments_by_queries_in_pksg_wrapper] query_path is not valid!')

    l_queries = []
    with open(query_path, 'r') as in_fd:
        for ln in in_fd:
            l_queries.append(ln.strip())
        in_fd.close()
    if len(l_queries) <= 0:
        raise Exception('[sentiments_by_queries_in_pksg_wrapper] l_queries is empty!')

    df_ph_str_to_ph_id = pd.read_pickle(global_settings.g_phrase_str_to_phrase_id_file_fmt.format(ph_ds_name))
    logging.debug('[sentiments_by_queries_in_pksg_wrapper] Load in %s recs for df_ph_str_to_ph_id'
                  % str(len(df_ph_str_to_ph_id)))

    l_query_ids = []
    for query in l_queries:
        if query in df_ph_str_to_ph_id.index:
            l_query_ids.append(df_ph_str_to_ph_id.loc[query][global_settings.g_ph_id_col])
    if len(l_query_ids) <= 0:
        logging.error('[sentiments_by_queries_in_pksg_wrapper] Cannot find queries in stored phrases!')
        return

    multitasking(single_task_func=sentiments_by_queries_in_pksg_single_task,
                 single_task_params=(l_query_ids, query_name, agg_type),
                 num_task=num_task,
                 l_task_ids=None,
                 job_id=job_id,
                 task_type='proc',
                 merge_int=False,
                 int_folder=None,
                 int_fmt=None,
                 after_merge_func=None,
                 after_merge_func_params=None,
                 out_path=None,
                 index_col=None,
                 rm_int=False)

    logging.debug('[sentiments_by_queries_in_pksg_wrapper] All done in %s secs.' % str(time.time() - timer_start))


def merge_sent_ts_from_jobs(pksg_ts_ds_name_prefix, query_name, pksg_ts_job_cnt, output_ds_name,
                            quotient_name=None, ph_ds_name=None):
    """
    When 'quotient_name' is not None, for each time interval, all queries are considered equivalent, and a representative
    query (arbitrarily selected) will be the key for the 'd_sent' column, in which each element is a dict.
        :param is_quotient: 'True' -- consider all queries equivalent.
    """
    logging.debug('[merge_sent_ts_from_jobs] Starts.')
    timer_start = time.time()

    if ph_ds_name is not None:
        df_ph_id_to_ph_str = pd.read_pickle(global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ph_ds_name))
        logging.debug('[merge_sent_ts_from_jobs] Load in df_ph_id_to_ph_str with %s phrases.' % len(df_ph_id_to_ph_str))

    # sent_ts dataset is a subset of pksg_ts dataset.
    l_pksg_ts_ds_name = []
    for i in range(int(pksg_ts_job_cnt)):
        pksg_ts_ds_name = pksg_ts_ds_name_prefix + '#' + str(i)
        with open(global_settings.g_pksg_ts_ds_name_file_fmt.format(pksg_ts_ds_name), 'r') as in_fd:
            for ln in in_fd:
                l_pksg_ts_ds_name.append(ln.strip())
            in_fd.close()

    d_t_int = dict()
    for pksg_ts_ds_name in l_pksg_ts_ds_name:
        l_fields = pksg_ts_ds_name.split('@')
        t_int = l_fields[0]
        if t_int not in d_t_int:
            d_t_int[t_int] = [pksg_ts_ds_name]
        else:
            d_t_int[t_int].append(pksg_ts_ds_name)

    l_sent_ts_rec = []
    for t_int in d_t_int:
        d_sent_by_t_int = dict()
        for pksg_ts_ds_name in d_t_int[t_int]:
            sent_ts_file_path = global_settings.g_sent_ts_int_file_fmt.format(pksg_ts_ds_name + '#' + query_name)
            if not path.exists(sent_ts_file_path):
                continue
            df_sent = pd.read_pickle(sent_ts_file_path)
            if len(df_sent) <= 0:
                continue
            for query_ph_id, sent_rec in df_sent.iterrows():
                if quotient_name is not None:
                    if len(d_sent_by_t_int) <= 0:
                        d_sent_by_t_int[quotient_name] = [sent_rec['agg_sents'], sent_rec['eff_support'],
                                                          sent_rec['support'], [query_ph_id], None]
                    else:
                        d_sent_by_t_int[quotient_name][0] += sent_rec['agg_sents']
                        d_sent_by_t_int[quotient_name][1] += sent_rec['eff_support']
                        d_sent_by_t_int[quotient_name][2] += sent_rec['support']
                        d_sent_by_t_int[quotient_name][3].append(query_ph_id)
                else:
                    if query_ph_id not in d_sent_by_t_int:
                        d_sent_by_t_int[query_ph_id] = [sent_rec['agg_sents'], sent_rec['eff_support'],
                                                        sent_rec['support'], [query_ph_id], None]
                    else:
                        d_sent_by_t_int[query_ph_id][0] += sent_rec['agg_sents']
                        d_sent_by_t_int[query_ph_id][1] += sent_rec['eff_support']
                        d_sent_by_t_int[query_ph_id][2] += sent_rec['support']
            if quotient_name is not None:
                d_sent_by_t_int[quotient_name][3] = list(set(d_sent_by_t_int[quotient_name][3]))

        for query_ph_id in d_sent_by_t_int:
            if d_sent_by_t_int[query_ph_id][1] > 0:
                d_sent_by_t_int[query_ph_id][0] = d_sent_by_t_int[query_ph_id][0] / d_sent_by_t_int[query_ph_id][1]
        if ph_ds_name is not None:
            for query_ph_id in d_sent_by_t_int:
                d_sent_by_t_int[query_ph_id][4] = df_ph_id_to_ph_str.loc[df_ph_id_to_ph_str.index
                    .intersection(d_sent_by_t_int[query_ph_id][3])][global_settings.g_ph_str_col].to_list()
        l_sent_ts_rec.append((t_int, d_sent_by_t_int))
        logging.debug('[merge_sent_ts_from_jobs] t_int %s done.' % t_int)
    df_sent_ts = pd.DataFrame(l_sent_ts_rec, columns=['t_int', 'd_sent'])
    df_sent_ts = df_sent_ts.set_index('t_int')
    pd.to_pickle(df_sent_ts, global_settings.g_sent_ts_file_fmt.format(output_ds_name))
    logging.debug('[merge_sent_ts_from_jobs] All done in %s secs.' % str(time.time() - timer_start))


def draw_sent_ts(ds_name, query_name, ph_ds_name, show_img, save_img, quotient_name=None):
    logging.debug('[draw_sent_ts] Starts.')
    timer_start = time.time()

    df_sent_ts = pd.read_pickle(global_settings.g_sent_ts_file_fmt.format(ds_name + '#' + query_name))
    logging.debug('[draw_sent_ts] Load in %s sent_ts recs.' % str(len(df_sent_ts)))

    df_ph_id_to_ph_str = pd.read_pickle(global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ph_ds_name))
    logging.debug('[draw_sent_ts] Load in %s recs for df_ph_str_df_ph_id_to_ph_strto_ph_id'
                  % str(len(df_ph_id_to_ph_str)))

    if quotient_name is not None:
        logging.debug('[draw_sent_ts] Use quotient queries with name %s' % quotient_name)

    l_query_id = []
    l_query_str = []
    if quotient_name is not None:
        l_query_id = [quotient_name]
        l_query_str = [quotient_name]
    else:
        for t_int, d_sent_rec in df_sent_ts.iterrows():
            d_sent = d_sent_rec['d_sent']
            for query_ph_id in d_sent:
                if query_ph_id not in df_ph_id_to_ph_str.index:
                    raise Exception('[draw_sent_ts] query_ph_id %s is not in df_ph_id_to_ph_str.' % query_ph_id)
                query_str = df_ph_id_to_ph_str.loc[query_ph_id][global_settings.g_ph_str_col]
                if query_str not in l_query_str:
                    l_query_str.append(query_str)
                if query_ph_id not in l_query_id:
                    l_query_id.append(query_ph_id)

    l_t_int = []
    d_sent_ts = {query_str: [] for query_str in l_query_str}
    d_eff_support = {query_str: [] for query_str in l_query_str}
    d_support = {query_str: [] for query_str in l_query_str}
    for t_int, d_sent_rec in df_sent_ts.iterrows():
        l_t_int.append(t_int)
        d_sent = d_sent_rec['d_sent']
        if quotient_name is not None:
            if len(d_sent) <= 0:
                d_sent_ts[quotient_name].append(np.zeros(4))
                d_eff_support[quotient_name].append(0)
                d_support[quotient_name].append(0)
            else:
                d_sent_ts[quotient_name].append(d_sent[quotient_name][0])
                d_eff_support[quotient_name].append(d_sent[quotient_name][1])
                d_support[quotient_name].append(d_sent[quotient_name][2])
        else:
            for idx, query_ph_id in enumerate(l_query_id):
                query_str = l_query_str[idx]
                if query_ph_id not in d_sent:
                    d_sent_ts[query_str].append(np.zeros(4))
                    d_eff_support[query_str].append(0)
                    d_support[query_str].append(0)
                else:
                    d_sent_ts[query_str].append(d_sent[query_ph_id][0])
                    d_eff_support[query_str].append(d_sent[query_ph_id][1])
                    d_support[query_str].append(d_sent[query_ph_id][2])

    l_t_int = sorted(l_t_int)
    l_t_int = [t_int.split('#')[0] for t_int in l_t_int]

    img_width = len(l_t_int) * 0.1
    if img_width < 15:
        img_width = 15
    img_height = len(l_query_str) * 15
    fig, axes = plt.subplots(ncols=1, nrows=3 * len(l_query_id), figsize=(img_width, img_height))
    fig.suptitle('%s Sentiment Time Series' % ds_name.upper(), fontsize=24, fontweight='semibold')
    linewidth = 2
    very_neg_color = 'navy'
    neg_color = 'cornflowerblue'
    pos_color = 'lightcoral'
    very_pos_color = 'maroon'
    reg_1_color = 'tab:blue'
    reg_2_color = 'tab:orange'
    sub_fig_fontsize = 15

    idx = 0
    for query_str in l_query_str:
        axes[idx].grid(True)
        axes[idx].set_title('Sentiment Time Series: %s' % query_str, fontsize=sub_fig_fontsize, fontweight='semibold')
        l_x_ticks = [i for i in range(len(l_t_int))]
        # l_x_ticks_sample = [i for i in range(0, len(l_t_int), 4)]
        l_x_ticks_sample = l_x_ticks
        l_t_int_sample = [l_t_int[x] for x in l_x_ticks_sample]
        axes[idx].plot(l_x_ticks, [sent_vec[0] for sent_vec in d_sent_ts[query_str]], linewidth=linewidth, color=very_neg_color, label='very neg')
        axes[idx].plot(l_x_ticks, [sent_vec[1] for sent_vec in d_sent_ts[query_str]], linewidth=linewidth, color=neg_color, label='neg')
        axes[idx].plot(l_x_ticks, [sent_vec[2] for sent_vec in d_sent_ts[query_str]], linewidth=linewidth, color=pos_color, label='pos')
        axes[idx].plot(l_x_ticks, [sent_vec[3] for sent_vec in d_sent_ts[query_str]], linewidth=linewidth, color=very_pos_color, label='very pos')
        axes[idx].set_xticks(l_x_ticks_sample)
        axes[idx].set_xticklabels(l_t_int_sample, rotation=45, ha='right', rotation_mode="anchor")
        axes[idx].legend()
        idx += 1

        axes[idx].grid(True)
        axes[idx].set_title('Sentiment Support: %s' % query_str, fontsize=sub_fig_fontsize, fontweight='semibold')
        axes[idx].plot(l_x_ticks, d_eff_support[query_str], linewidth=linewidth, color=reg_1_color, label='sentiment support')
        axes[idx].set_xticks(l_x_ticks_sample)
        axes[idx].set_xticklabels(l_t_int_sample, rotation=45, ha='right', rotation_mode="anchor")
        max_y = max(d_eff_support[query_str])
        y_stride = math.floor(max_y / 8)
        if y_stride <= 1:
            y_stride = 1
        l_y_ticks = [y for y in range(0, max_y, y_stride)]
        axes[idx].set_yticks(l_y_ticks)
        axes[idx].set_yticklabels(l_y_ticks)
        axes[idx].legend()
        idx += 1

        axes[idx].grid(True)
        axes[idx].set_title('Total Support: %s' % query_str, fontsize=sub_fig_fontsize, fontweight='semibold')
        axes[idx].plot(l_x_ticks, d_support[query_str], linewidth=linewidth, color=reg_2_color, label='total support')
        axes[idx].set_xticks(l_x_ticks_sample)
        axes[idx].set_xticklabels(l_t_int_sample, rotation=45, ha='right', rotation_mode="anchor")
        max_y = max(d_support[query_str])
        y_stride = math.floor(max_y / 8)
        if y_stride <= 1:
            y_stride = 1
        l_y_ticks = [y for y in range(0, max_y, y_stride)]
        axes[idx].set_yticks(l_y_ticks)
        axes[idx].set_yticklabels(l_y_ticks)
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.94)

    if save_img:
        img_name = ds_name + '#' + query_name
        plt.savefig(global_settings.g_sent_ts_img_file_fmt.format(img_name), format='PNG')
    if show_img:
        plt.show()
    plt.clf()
    plt.close()
    logging.debug('[draw_sent_ts] All done in %s secs.' % str(time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        # gen_tasks 202005 2 2 10
        pksg_ts_ds_name_prefix = sys.argv[2]
        pksg_ts_job_cnt = sys.argv[3]
        trg_job_cnt = sys.argv[4]
        task_cnt_per_job = sys.argv[5]
        gen_sent_ts_tasks(pksg_ts_ds_name_prefix, pksg_ts_job_cnt, trg_job_cnt, task_cnt_per_job)
    elif cmd == 'sent_ts':
        # sent_ts 10 0 ca covid
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        ph_ds_name = sys.argv[4]
        query_ds_name = sys.argv[5]
        sentiments_by_queries_in_pksg_wrapper(num_task, job_id, ph_ds_name, query_ds_name,
                                              scenario_settings.g_query_file_fmt.format(query_ds_name))
    elif cmd == 'merge_sent_ts':
        # merge_sent_ts ca 2 1
        ds_name = sys.argv[2]
        query_name = sys.argv[3]
        pksg_ts_job_cnt = sys.argv[4]
        if len(sys.argv) >= 6 or sys.argv[5].strip() == 'None':
            quotient_name = sys.argv[5]
        else:
            quotient_name = None
        if len(sys.argv) >= 7:
            ph_ds_name = sys.argv[6]
        else:
            ph_ds_name = None
        pksg_ts_ds_name_prefix = ds_name
        output_ds_name = ds_name + '#' + query_name
        merge_sent_ts_from_jobs(pksg_ts_ds_name_prefix, query_name, pksg_ts_job_cnt, output_ds_name, quotient_name,
                                ph_ds_name)
    elif cmd == 'draw_sent_ts':
        # draw_sent_ts ca 0 1 "covid_19_vaccination"
        plt.set_loglevel('error')
        ds_name = sys.argv[2]
        query_name = sys.argv[3]
        show_img = bool(int(sys.argv[4]))
        save_img = bool(int(sys.argv[5]))
        if len(sys.argv) >= 7 and sys.argv[6] != '':
            quotien_name = sys.argv[6]
        else:
            quotien_name = None
        ph_ds_name = ds_name
        draw_sent_ts(ds_name, query_name, ph_ds_name, show_img, save_img, quotien_name)
