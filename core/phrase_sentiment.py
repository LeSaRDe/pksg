import logging
import time

import numpy as np
import pandas as pd

from util import global_settings
from util.multitasking import multitasking, merge_int_rets

"""
PKSG PIPELINE
STAGE: PHRASE SENTIMENT (PH_SENT)

REQUIREMENTS:
    SGRAPH, TXT_PH

OUTPUTS:
    text->phrase sentiments table: g_phrase_sent_file_fmt
        pandas DataFrame
        Index: hash text (int): g_hash_txt_col
        Columns: phrase sentiment tuples (list of tuples): g_phrase_sent_file_fmt
        
        Each phrase sentiment tuple is 
        ([token_1, token_2], 
         [pos_1, pos_2], 
         [(token_1_start, token_1_end), (token_2_start, token_2_end)],
         phrase_sentiment)
"""


def sentiment_calibration(sent_vec_1, sent_vec_2):
    '''
    Return: (True for calibrated, sent_vec)
    '''
    sent_vec_1 = np.asarray(sent_vec_1)
    sent_vec_2 = np.asarray(sent_vec_2)
    sent_class_1 = np.abs(np.argmax(sent_vec_1) - 2)
    sent_class_2 = np.abs(np.argmax(sent_vec_2) - 2)
    if sent_class_1 > sent_class_2:
        return True, sent_vec_1
    elif sent_class_2 > sent_class_1:
        return True, sent_vec_2
    else:
        return False, (sent_vec_1 + sent_vec_2) / 2.0


def find_mininal_subtree_from_specified_root_for_phrase(sgraph, root, phrase_start, phrase_end):
    cur_node = (root, sgraph.nodes(data=True)[root])
    if phrase_start < cur_node[1]['token_start'] or phrase_end > cur_node[1]['token_end']:
        return None
    while True:
        if phrase_start >= cur_node[1]['token_start'] and phrase_end <= cur_node[1]['token_end']:
            go_deep = False
            for child in sgraph.successors(cur_node[0]):
                if phrase_start >= sgraph.nodes(data=True)[child]['token_start'] \
                        and phrase_end <= sgraph.nodes(data=True)[child]['token_end']:
                    cur_node = (child, sgraph.nodes(data=True)[child])
                    go_deep = True
                    break
            if go_deep:
                continue
            else:
                break
    return cur_node[0]


def compute_phrase_sentiment(sgraph, phrase_start, phrase_end, l_span):
    '''
    'phrase_start' and 'phrase_end' are the exact and absolute character indices of a phrase relative to the text.
    '''
    # find the minimal segment in sgraph that includes [phrase_start, phrase_end]
    root = None
    for node in sgraph.nodes():
        if sgraph.in_degree(node) == 0:
            root = node
    if root is None:
        raise Exception('[compute_phrase_sentiment] No root found.')

    phrase_sentiment = None
    min_subtree_root = find_mininal_subtree_from_specified_root_for_phrase(sgraph, root, phrase_start, phrase_end)
    if min_subtree_root is not None:
        phrase_sentiment = sgraph.nodes(data=True)[min_subtree_root]['sentiments']
    if phrase_sentiment is None:
        raise Exception('[compute_phrase_sentiment] Cannot get sentiment for phrase: %s, %s.'
                        % (phrase_start, phrase_end))
    elif len(l_span) > 1:
        # sentiment calibration:
        # when the input phrase as a whole is determined to be neutral, we look into the sentiment of each individual
        # token, and prefer to emphasize the most polarized sentiment.
        # in the case that one token is positive and the other is negative, we still prefer to emphasized the most
        # polarized one.
        primary_sentiment_class = np.argmax(phrase_sentiment)
        if primary_sentiment_class == 2:
            token_1_start = l_span[0][0]
            token_1_end = l_span[0][1]
            token_2_start = l_span[1][0]
            token_2_end = l_span[1][1]
            subtree_root_1 = min_subtree_root
            subtree_root_2 = min_subtree_root
            min_subtree_root_1 = find_mininal_subtree_from_specified_root_for_phrase(sgraph, subtree_root_1,
                                                                                     token_1_start, token_1_end)
            min_subtree_root_2 = find_mininal_subtree_from_specified_root_for_phrase(sgraph, subtree_root_2,
                                                                                     token_2_start, token_2_end)
            if min_subtree_root_1 is not None and min_subtree_root_2 is not None:
                token_1_sentiment = sgraph.nodes(data=True)[min_subtree_root_1]['sentiments']
                token_2_sentiment = sgraph.nodes(data=True)[min_subtree_root_2]['sentiments']
                if token_1_sentiment is None or token_2_sentiment is None:
                    raise Exception('[compute_phrase_sentiment] Something wrong when getting sentiments for %s.'
                                    % str(l_span))
                # TODO
                # may need a better function calibrates the sentiment based on token_1_sentiment and token_2_sentiment
                # token_1_sentiment_class = np.abs(np.argmax(token_1_sentiment) - 2)
                # token_2_sentiment_class = np.abs(np.argmax(token_2_sentiment) - 2)
                # if token_1_sentiment_class == 0 and token_2_sentiment_class == 0:
                #     return phrase_sentiment
                # elif token_1_sentiment_class > token_2_sentiment_class:
                #     return token_1_sentiment
                # elif token_2_sentiment_class > token_1_sentiment_class:
                #     return token_2_sentiment
                # else:
                #     return ((np.asarray(token_1_sentiment) + np.asarray(token_2_sentiment)) / 2.0).tolist()
                calibrated, calibrated_phrase_sentiment = sentiment_calibration(token_1_sentiment, token_2_sentiment)
                if calibrated:
                    return calibrated_phrase_sentiment
                else:
                    return phrase_sentiment
            else:
                logging.error('[compute_phrase_sentiment] Something wrong sentiment calibration: sgraph: %s, '
                              'phrase_start: %s, phrase_end: %s, l_span: %s' % sgraph, phrase_start, phrase_end, l_span)
                return phrase_sentiment
        else:
            return phrase_sentiment
    else:
        return phrase_sentiment


def compute_sentiment_for_one_phrase_in_one_txt(l_parsed_sgraph_info, phrase_start, phrase_end, l_phspan):
    '''
    'l_parsed_sgraph_info' is the list of sgraphs for the given tweet.
    Each element is a tuple: (sgraph in nx.Graph, sgraph_start, sgraph_end)
    'l_phspan' is for the given phrase. Each element is for a token in the phrase.
    '''
    phrase_sentiment = None
    for sgraph_info in l_parsed_sgraph_info:
        sgraph = sgraph_info[0]
        sgraph_start = sgraph_info[1]
        sgraph_end = sgraph_info[2]
        if phrase_start >= sgraph_start and phrase_end <= sgraph_end:
            phrase_sentiment = compute_phrase_sentiment(sgraph, phrase_start, phrase_end, l_phspan)
            break
        else:
            continue
    return phrase_sentiment


def compute_phrase_sentiment_for_one_txt(l_sgraph, l_phrase):
    '''
    'l_sgraph_info' and 'l_phrase' are for one tweet.
    Returns [([token_1, token_2], [pos_1, pos_2], [(token_1_start, token_1_end), (token_2_start, token_2_end)],
    phrase_sentiment), ...]
    '''
    if l_sgraph is None or len(l_sgraph) <= 0 or l_phrase is None or len(l_phrase) <= 0:
        logging.error('[compute_phrase_sentiment_for_one_txt] Invalid l_sgraph_info or l_phrase.')
        return None

    l_parsed_sgraph_info = []
    for sgraph_info in l_sgraph:
        sgraph = sgraph_info[0]
        sgraph_start = sgraph_info[1]
        sgraph_end = sgraph_info[2]
        l_parsed_sgraph_info.append((sgraph, sgraph_start, sgraph_end))

    l_ready = []
    l_leftover = []
    for phrase in l_phrase:
        l_token = phrase[0]
        l_pos = phrase[1]
        l_span = phrase[2]
        if len(l_token) == 1:
            phrase_start = l_span[0][0]
            phrase_end = l_span[0][1]
        elif len(l_token) == 2:
            phrase_token_1_start = l_span[0][0]
            phrase_token_1_end = l_span[0][1]
            phrase_token_2_start = l_span[1][0]
            phrase_token_2_end = l_span[1][1]
            phrase_start = min(phrase_token_1_start, phrase_token_2_start)
            phrase_end = max(phrase_token_1_end, phrase_token_2_end)
        else:
            raise Exception('[compute_phrase_sentiment_for_one_txt] Invalid phrase: %s' % str(phrase))

        phrase_sentiment = compute_sentiment_for_one_phrase_in_one_txt(l_parsed_sgraph_info, phrase_start, phrase_end,
                                                                       l_span)
        if phrase_sentiment is None:
            l_leftover.append(phrase)
        l_ready.append((l_token, l_pos, l_span, phrase_sentiment))
    return l_ready, l_leftover


def phrase_sent_single_task(task_id, hash_txt_col, phrase_col, sgraph_col, phrase_sent_col, leftover_col):
    logging.debug('[phrase_sent_single_task] Task %s: Starts.' % str(task_id))
    timer_start = time.time()

    df_task = pd.read_pickle(global_settings.g_phrase_sent_task_file_fmt.format(task_id))
    logging.debug('[phrase_sent_single_task] Task %s: Load in %s task recs.' % (task_id, len(df_task)))

    read_cnt = 0
    l_ready = []
    leftover_phrase_cnt = 0
    l_leftover_rec = []
    for hash_txt, task_rec in df_task.iterrows():
        l_phrase = task_rec[phrase_col]
        l_sgraph = task_rec[sgraph_col]
        l_phrase_sentiment, l_leftover = compute_phrase_sentiment_for_one_txt(l_sgraph, l_phrase)
        l_ready.append((hash_txt, l_phrase_sentiment))
        if len(l_leftover) > 0:
            l_leftover_rec.append((hash_txt, l_leftover))
            leftover_phrase_cnt += len(l_leftover)
        read_cnt += 1
        if read_cnt % 5000 == 0 and read_cnt >= 5000:
            logging.debug('[phrase_sent_single_task] Task %s: %s recs done in %s secs. '
                          'leftover_phrase_cnt: %s'
                          % (task_id, read_cnt, time.time() - timer_start, leftover_phrase_cnt))
    logging.debug('[phrase_sent_single_task] Task %s: %s recs done in %s secs. '
                  'leftover_phrase_cnt: %s'
                  % (task_id, read_cnt, time.time() - timer_start, leftover_phrase_cnt))
    df_phrase_sentiment = pd.DataFrame(l_ready, columns=[hash_txt_col, phrase_sent_col])
    pd.to_pickle(df_phrase_sentiment, global_settings.g_phrase_sent_int_file_fmt.format(task_id))
    df_leftover = pd.DataFrame(l_leftover_rec, columns=[hash_txt_col, leftover_col])
    pd.to_pickle(df_leftover, global_settings.g_phrase_sent_leftover_int_file_fmt.format(task_id))
    logging.debug('[phrase_sent_single_task] Task %s: All done with %s recs and %s leftover in %s secs.'
                  % (task_id, len(df_phrase_sentiment), len(df_leftover), time.time() - timer_start))


def merge_leftover_int_rets(df_merge, job_id, ds_name, index_col):
    '''
    Merge leftover int rets without changing anything on df_merge.
    '''
    df_leftover = merge_int_rets(global_settings.g_phrase_sent_int_folder,
                                 global_settings.g_phrase_sent_leftover_int_fmt_regex.format(str(job_id)),
                                 index_col,
                                 False)
    if df_leftover is not None:
        pd.to_pickle(df_leftover, global_settings.g_phrase_sent_leftover_file_fmt.format(ds_name))
    return df_merge


def phrase_sent_wrapper(ds_name, num_task, job_id, index_col=global_settings.g_hash_txt_col,
                        phrase_col=global_settings.g_txt_phrase_col,
                        sgraph_col=global_settings.g_sgraph_col,
                        phrase_sent_col=global_settings.g_ph_sent_col,
                        leftover_col=global_settings.g_ph_sent_leftover_col):
    logging.debug('[phrase_sent_wrapper] Starts')
    timer_start = time.time()

    multitasking(single_task_func=phrase_sent_single_task,
                 single_task_params=(index_col, phrase_col, sgraph_col, phrase_sent_col, leftover_col),
                 num_task=num_task,
                 l_task_ids=None,
                 job_id=job_id,
                 task_type='proc',
                 int_folder=global_settings.g_phrase_sent_int_folder,
                 int_fmt=global_settings.g_phrase_sent_int_fmt_regex.format(str(job_id)),
                 after_merge_func=merge_leftover_int_rets,
                 after_merge_func_params=(job_id, ds_name, index_col),
                 out_path=global_settings.g_phrase_sent_file_fmt.format(ds_name),
                 index_col=index_col,
                 rm_int=False)

    logging.debug('[phrase_sent_wrapper] All done in %s secs.' % str(time.time() - timer_start))
