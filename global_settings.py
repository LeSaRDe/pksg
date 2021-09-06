"""
INPUTS
"""
g_src_folder = '/home/mf3jh/workspace/pksg/'

g_local_or_remote = 'local'

if g_local_or_remote == 'remote':
    # RIVANNA
    g_work_folder = '/scratch/mf3jh/data/covid/'
    # g_work_folder = '/scratch/mf3jh/data/ven_tw_pksg/'
    # g_raw_data_folder = g_work_folder + 'tw_raw_data/carley/'
    g_raw_data_folder = g_work_folder + 'tw_raw_data/va_vax2021_jan_may/'
    # g_raw_data_folder = '/scratch/mf3jh/data/cp4_ven_tw/tw_raw_data/'
    g_conf_folder = g_work_folder + 'conf/'
else:
    # LOCAL
    # g_work_folder = '/home/mf3jh/workspace/data/covid/'
    g_work_folder = '/home/mf3jh/workspace/data/aspect_sentiment_analysis/'
    # g_raw_data_folder = g_work_folder + 'tw_raw_data/carley/'
    # g_raw_data_folder = g_work_folder + 'tw_raw_data/2021_covid_geo_tweets/'
    # g_raw_data_folder = g_work_folder + 'tw_raw_data/202001_sample/'
    # g_raw_data_folder = g_work_folder + 'tw_raw_data/202005/'
    # g_raw_data_folder = g_work_folder + 'tw_raw_data/pure_test/'
    # g_raw_data_folder = g_work_folder + 'tw_raw_data/va_vax2021_jan_may/'
    g_raw_data_folder = g_work_folder + 'raw_data/'
    g_conf_folder = g_work_folder + 'conf/'

"""
GENERATEDs
"""
# RAW_TXT
g_raw_txt_info_folder = g_work_folder + 'raw_txt_info/'
g_raw_txt_info_int_folder = g_raw_txt_info_folder + 'int/'
g_raw_txt_info_int_file_fmt = g_raw_txt_info_int_folder + 'raw_txt_info_int_{0}.pickle'
g_raw_txt_info_int_fmt_regex = r'raw_txt_info_int_.*\.pickle'
g_raw_txt_info_task_file_fmt = g_raw_txt_info_int_folder + 'raw_txt_info_task_{0}.txt'
g_raw_txt_info_file_fmt = g_raw_txt_info_folder + 'raw_txt_info_{0}.pickle'
g_txt_id_to_ref_file_fmt = g_raw_txt_info_folder + 'txt_id_to_ref_{0}.pickle'

# TXT_CLEAN
g_txt_clean_folder = g_work_folder + 'txt_clean/'
g_txt_clean_int_folder = g_txt_clean_folder + 'int/'
g_txt_clean_task_file_fmt = g_txt_clean_int_folder + 'txt_clean_task_{0}.pickle'
g_txt_clean_int_file_fmt = g_txt_clean_int_folder + 'txt_clean_int_{0}.pickle'
g_txt_clean_int_fmt_regex = r'txt_clean_int_.*\.pickle'
g_txt_clean_file_fmt = g_txt_clean_folder + 'clean_txt_{0}.pickle'

# TXT_HASH
g_txt_hash_folder = g_work_folder + 'txt_hash/'
g_txt_hash_int_folder = g_txt_hash_folder + 'int/'
g_txt_hash_task_file_fmt = g_txt_hash_int_folder + 'txt_hash_task_{0}.pickle'
g_txt_hash_int_file_fmt = g_txt_hash_int_folder + 'txt_hash_int_{0}.pickle'
g_txt_hash_int_fmt_regex = r'txt_hash_int_.*\.pickle'
g_txt_id_to_hash_txt_file_fmt = g_txt_hash_folder + 'txt_id_to_hash_txt_{0}.pickle'
g_txt_hash_file_fmt = g_txt_hash_folder + 'hash_txt_{0}.pickle'

# SGRAPH
g_txt_sent_folder = g_work_folder + 'sent_graph/'
g_txt_sent_int_folder = g_txt_sent_folder + 'int/'
g_txt_sent_task_file_fmt = g_txt_sent_int_folder + 'txt_sent_task_{0}.json'
g_txt_sent_int_file_fmt = g_txt_sent_int_folder + 'txt_sent_int_{0}.json'
g_sgraph_int_file_fmt = g_txt_sent_int_folder + 'sgraph_int_{0}.pickle'
g_sgraph_int_fmt_regex = r'sgraph_int_{0}#.*\.pickle'
g_sgraph_file_fmt = g_txt_sent_folder + 'sgraph_{0}.pickle'

# SEM_UNIT
g_sem_unit_folder = g_work_folder + 'sem_unit/'
g_sem_unit_int_folder = g_sem_unit_folder + 'int/'
g_sem_unit_task_file_fmt = g_sem_unit_int_folder + 'sem_unit_task_{0}.pickle'
g_sem_unit_int_file_fmt = g_sem_unit_int_folder + 'sem_unit_int_{0}.pickle'
g_sem_unit_int_fmt_regex = r'sem_unit_int_.*\.pickle'
g_sem_unit_file_fmt = g_sem_unit_folder + 'sem_unit_{0}.pickle'

# TXT_PH
g_txt_phrase_folder = g_work_folder + 'txt_phrase/'
g_txt_phrase_int_folder = g_txt_phrase_folder + 'int/'
g_txt_phrase_task_file_fmt = g_txt_phrase_int_folder + 'txt_phrase_task_{0}.pickle'
g_txt_phrase_int_file_fmt = g_txt_phrase_int_folder + 'txt_phrase_int_{0}.pickle'
g_txt_phrase_int_fmt_regex = r'txt_phrase_int_.*\.pickle'
g_txt_phrase_file_fmt = g_txt_phrase_folder + 'txt_phrase_{0}.pickle'
g_hash_txt_to_ph_ids_file_fmt = g_txt_phrase_folder + 'hash_txt_to_ph_ids_{0}.pickle'
g_ph_id_to_hash_txts_file_fmt = g_txt_phrase_folder + 'ph_id_to_hash_txts_{0}.pickle'
g_phrase_id_to_phrase_str_file_fmt = g_txt_phrase_folder + 'ph_id_to_ph_str_{0}.pickle'
g_phrase_str_to_phrase_id_file_fmt = g_txt_phrase_folder + 'ph_str_to_ph_id_{0}.pickle'
g_token_filter_file = g_txt_phrase_folder + 'token_filter.txt'

# PH_SENT
g_phrase_sent_folder = g_work_folder + 'phrase_sent/'
g_phrase_sent_int_folder = g_phrase_sent_folder + 'int/'
g_phrase_sent_task_file_fmt = g_phrase_sent_int_folder + 'ph_sent_task_{0}.pickle'
g_phrase_sent_int_file_fmt = g_phrase_sent_int_folder + 'ph_sent_int_{0}.pickle'
g_phrase_sent_int_fmt_regex = r'ph_sent_int_{0}#.*\.pickle'
g_phrase_sent_file_fmt = g_phrase_sent_folder + 'ph_sent_{0}.pickle'
g_phrase_sent_leftover_int_file_fmt = g_phrase_sent_int_folder + 'ph_sent_leftover_int_{0}.pickle'
g_phrase_sent_leftover_int_fmt_regex = r'ph_sent_leftover_int_{0}#.*\.pickle'
g_phrase_sent_leftover_file_fmt = g_phrase_sent_folder + 'ph_sent_leftover_{0}.pickle'

# phrase embedding
g_phrase_embed_folder = g_work_folder + 'phrase_embed/'
g_phrase_embed_int_folder = g_phrase_embed_folder + 'int/'
g_phrase_embed_task_file_fmt = g_phrase_embed_int_folder + 'phrase_embed_task_{0}.pickle'
g_phrase_embed_int_file_fmt = g_phrase_embed_int_folder + 'phrase_embed_int_{0}.pickle'
g_phrase_embed_file_fmt = g_phrase_embed_folder + 'phrase_embed_{0}.pickle'
# g_tw_to_phrase_id_int_file_fmt = g_phrase_embed_int_folder + 'tw_to_phrase_id_int_{0}.pickle'
g_phrase_row_id_to_phrase_id_file_fmt = g_phrase_embed_folder + 'phrase_row_id_to_phrase_id_{0}.json'

g_token_embed_task_file_fmt = g_phrase_embed_int_folder + 'token_embed_task_{0}.pickle'
g_token_embed_int_file_fmt = g_phrase_embed_int_folder + 'token_embed_int_{0}.pickle'
g_token_embed_file_fmt = g_phrase_embed_folder + 'token_embed_{0}.pickle'
g_token_embed_combined_file_fmt = g_phrase_embed_folder + 'token_embed_combined_{0}.pickle'

# phrase clustering
g_phrase_cluster_folder = g_work_folder + 'phrase_cluster/'
g_phrase_cluster_int_folder = g_phrase_cluster_folder + 'int/'
g_phrase_cluster_task_file_fmt = g_phrase_cluster_int_folder + 'phrase_cluster_task_{0}.pickle'
g_phrase_cluster_int_file_fmt = g_phrase_cluster_int_folder + 'phrase_cluster_int_{0}.pickle'
g_phrase_cluster_pid_to_cid_int_file_fmt = g_phrase_cluster_int_folder + 'pid_to_cid_int_{0}.pickle'
g_phrase_sim_task_file_fmt = g_phrase_cluster_int_folder + 'phrase_sim_task_{0}.txt'
g_phrase_sim_row_id_to_phrase_id_file_fmt = g_phrase_cluster_folder + 'phrase_sim_row_id_to_phrase_id_{0}.json'
g_phrase_sim_int_file_fmt = g_phrase_cluster_int_folder + 'phrase_sim_int_{0}.pickle'
g_phrase_sim_graph_int_file_fmt = g_phrase_cluster_int_folder + 'phrase_sim_graph_int_{0}.pickle'
g_phrase_sim_graph_adj_int_file_fmt = g_phrase_cluster_int_folder + 'phrase_sim_graph_adj_int_{0}.npz'
g_phrase_sim_graph_adj_file_fmt = g_phrase_cluster_folder + 'phrase_sim_graph_adj_{0}.npz'
g_phrase_sim_graph_file = g_phrase_cluster_int_folder + 'phrase_sim_graph_{0}.json'
g_phrase_cluster_file_fmt = g_phrase_cluster_folder + 'phrase_cluster_{0}.pickle'

# knowledge graph
g_kgraph_folder = g_work_folder + 'kgraph/'
g_kgraph_int_folder = g_kgraph_folder + 'int/'
g_ks_graph_query_to_phrases_file_fmt = g_kgraph_folder + 'query_to_phrases_{0}.pickle'
g_ks_graph_q_phrase_to_sim_phrase_file_fmt = g_kgraph_folder + 'q_phrase_to_sim_phrase_{0}.pickle'
g_ks_graph_task_file_fmt = g_kgraph_int_folder + 'ks_graph_task_{0}.pickle'
g_ks_graph_int_file_fmt = g_kgraph_int_folder + 'ks_graph_int_{0}.pickle'
g_ks_graph_file_fmt = g_kgraph_folder + 'ks_graph_{0}.pickle'
g_phrase_cluster_id_to_phrase_id_file_fmt = g_kgraph_int_folder + 'pcid_to_pid_{0}.pickle'
g_phrase_id_to_phrase_cluster_id_file_fmt = g_kgraph_int_folder + 'pid_to_pcid_{0}.pickle'
g_ks_ctr_graph_int_file_fmt = g_kgraph_int_folder + 'ks_ctr_graph_int_{0}.pickle'

# PKSG
g_pksg_folder = g_work_folder + 'pksg/'
g_pksg_int_folder = g_pksg_folder + 'int/'
g_pksg_task_file_fmt = g_pksg_int_folder + 'pksg_task_{0}.txt'
g_pksg_int_file_fmt = g_pksg_int_folder + 'pksg_int_{0}.pickle'
g_pksg_int_fmt_regex = r'pksg_int_{0}#.*\.pickle'
g_pksg_file_fmt = g_pksg_folder + 'pksg_{0}.pickle'

# MERGE_PKSG
g_merge_pksg_folder = g_work_folder + 'merge_pksg/'
g_merge_pksg_int_folder = g_merge_pksg_folder + 'int/'
g_merge_pksg_task_file_fmt = g_merge_pksg_int_folder + 'merge_pksg_task_{0}.pickle'
g_merge_pksg_int_file_fmt = g_merge_pksg_int_folder + 'merge_pksg_int_{0}.pickle'
g_merge_pksg_int_fmt_regex = r'merge_pksg_int_.*\.pickle'
g_merge_pksg_file_fmt = g_merge_pksg_folder + 'merge_pksg_{0}.pickle'
g_merge_pksg_ts_task_file_fmt = g_merge_pksg_int_folder + 'merge_pksg_ts_task_{0}.pickle'
g_merge_pksg_ts_int_file_fmt = g_merge_pksg_int_folder + 'merge_pksg_ts_int_{0}.pickle'
g_merge_pksg_ts_int_fmt_regex = r'merge_pksg_ts_int_.*\.pickle'
g_merge_pksg_ts_file_fmt = g_merge_pksg_folder + 'merge_pksg_ts_{0}.pickle'

g_tw_tkg_task_file_fmt = g_kgraph_int_folder + 'tw_tkg_task_{0}.txt'
g_tw_tkg_int_file_fmt = g_kgraph_int_folder + 'tw_tkg_int_{0}.pickle'
g_merged_tw_tkg_task_file_fmt = g_kgraph_int_folder + 'merged_tw_tkg_task_{0}.pickle'
g_merged_tw_tkg_int_file_fmt = g_kgraph_int_folder + 'merged_tw_tkg_int_{0}.pickle'
g_merged_tw_tkg_file_fmt = g_kgraph_folder + 'merged_tw_tkg_{0}.pickle'

# FILTER_PKSG
g_filter_pksg_folder = g_work_folder + 'filter_pksg/'
g_filter_pksg_int_folder = g_filter_pksg_folder + 'int/'
# g_filter_pksg_task_file_fmt = g_filter_pksg_int_folder + 'filter_pksg_task_{0}.pickle'
# g_filter_pksg_int_filt_fmt = g_filter_pksg_int_folder + 'filter_pksg_int_{0}.pickle'
# g_filter_pksg_int_fmt_regex = r'filter_pksg_int_{0}.*\.pickle'

g_filter_node_task_file_fmt = g_filter_pksg_int_folder + 'filter_node_task_{0}.pickle'
g_filter_node_int_file_fmt = g_filter_pksg_int_folder + 'filter_node_int_{0}.pickle'
g_filter_node_int_fmt_regex = r'filter_node_int_{0}.*\.pickle'
g_filtered_nodes_file_fmt = g_filter_pksg_folder + 'filtered_nodes_{0}.txt'

g_filter_edge_task_file_fmt = g_filter_pksg_int_folder + 'filter_edge_task_{0}.pickle'
g_filter_edge_int_file_fmt = g_filter_pksg_int_folder + 'filter_edge_int_{0}.pickle'
g_filter_edge_int_fmt = r'filter_edge_int_{0}.*\.pickle'
g_filtered_edges_file_fmt = g_filter_pksg_folder + 'filtered_edges_{0}.pickle'

g_filtered_pksg_file_fmt = g_filter_pksg_folder + 'filtered_pksg_{0}.pickle'

# T_INT
g_t_int_folder = g_work_folder + 't_int/'
g_t_int_file_fmt = g_t_int_folder + 't_int_{0}.pickle'
g_pksg_ts_folder = g_t_int_folder + 'pksg_ts/'
g_pksg_ts_file_fmt = g_pksg_ts_folder + 'pksg_ts_{0}.pickle'
g_pksg_ts_fmt_regex = r'pksg_ts_.+@.+#.+.pickle'
g_pksg_ts_ds_name_file_fmt = g_pksg_ts_folder + 'pksg_ts_ds_name_{0}.txt'

# DATA_VIZ
g_data_viz_folder = g_work_folder + 'data_viz/'

# SENT_TS
g_sent_ts_folder = g_work_folder + 'sent_ts/'
g_sent_ts_int_folder = g_sent_ts_folder + 'int/'
g_sent_ts_task_file_fmt = g_sent_ts_int_folder + 'sent_ts_task_{0}.txt'
g_sent_ts_int_file_fmt = g_sent_ts_int_folder + 'sent_ts_int_{0}.pickle'
g_sent_ts_file_fmt = g_sent_ts_folder + 'sent_ts_{0}.pickle'
g_sent_ts_img_file_fmt = g_sent_ts_folder + 'sent_ts_img_{0}.png'

# USR_PRJ
g_usr_prj_folder = g_work_folder + 'usr_prj/'
g_usr_prj_int_folder = g_usr_prj_folder + 'int/'
g_rtug_ts_task_file_fmt = g_usr_prj_int_folder + 'rtug_task_{0}.pickle'
g_rtug_ts_int_file_fmt = g_usr_prj_int_folder + 'rtug_int_{0}.pickle'
g_rtug_ts_int_fmt_regex = r'rtug_int_{0}.*\.pickle'
g_rtug_ts_file_fmt = g_usr_prj_folder + 'rtug_{0}.pickle'


# embedding adjustment
g_adj_embed_folder = g_work_folder + 'adj_embed/'
g_adj_embed_file_fmt = g_adj_embed_folder + 'adj_ph_embed_{0}.pt'
g_adj_embed_dist_file_fmt = g_adj_embed_folder + 'adj_ph_embed_dist_{0}.npy'
g_adj_embed_samples_file_fmt = g_adj_embed_folder + 'adj_ph_embed_samples_{0}.pickle'

g_adj_token_embed_file_fmt = g_adj_embed_folder + 'adj_token_embed_{0}.pickle'

# curvature
g_curvature_folder = g_work_folder + 'curvature/'
g_curvature_int_folder = g_curvature_folder + 'int/'
g_kstest_task_file_fmt = g_curvature_int_folder + 'kstest_task_{0}.pickle'
g_kstest_int_file_fmt = g_curvature_int_folder + 'kstest_int_{0}.pickle'

# evaluation
g_eval_folder = g_work_folder + 'eval/'
g_token_list_file = g_eval_folder + 'tokens.txt'
g_ordered_token_list_file = g_eval_folder + 'ordered_tokens.txt'
g_token_embed_collect_file_fmt = g_eval_folder + 'token_embed_collect_{0}.pickle'
g_fixed_points_file = g_eval_folder + 'fp.pickle'
g_fixed_points_by_deg_file = g_eval_folder + 'fp_by_deg.pickle'
g_shared_fixed_points_by_deg_file = g_eval_folder + 'shared_fp_by_deg.txt'

g_intersect_tkg_file_fmt = g_eval_folder + 'intersect_tkg_{0}.pickle'

# test only
g_test_folder = g_work_folder + 'test/'
g_test_tkg = g_test_folder + 'tkg_test.pickle'
g_test_fp = g_test_folder + 'fp_test.txt'
g_test_orig_token_embed = g_test_folder + 'orig_token_embed_test.pickle'


"""
DATA STRUCTURES
"""
# shared by all
g_txt_idx = 'txt_id'

# RAW_TXT
g_raw_txt_col = 'raw_txt'
g_referee_txt_id = 'referee_txt_id'
g_referer_txt_ids = 'referer_txt_ids'

# TXT_CLEAN
g_clean_txt_col = 'clean_txt'

# TXT_HASH
g_hash_txt_col = 'hash_txt'

# SEM_UNIT
g_sem_unit_cls_col = 'cls_graph'
g_sem_unit_nps_col = 'nps'

# TXT_PH
g_txt_phrase_col = 'phrase'
g_ph_id_col = 'ph_id'
g_ph_str_col = 'ph_str'

# SGRAPH
g_sgraph_col = 'sgraph'

# PH_SENT
g_ph_sent_col = 'ph_sent'
g_ph_sent_leftover_col = 'ph_sent_leftover'

# PKSG
g_pksg_col = 'pksg'

# MERGE_PKSG
g_merge_pksg_col = 'merge_pksg'

# T_INT
g_t_int_id = 't_int_id'
g_t_int_start_col = 't_int_start'
g_t_int_end_col = 't_int_end'
g_t_int_txt_ids = 't_int_txt_ids'
# 't_int_name' is of the format: YYYYMMDDhhmmss#YYYYMMDDhhmmss
g_t_int_name = 't_int_name'

# FILTER_PKSG
g_node_id_col = 'node_id'
g_edge_id_col = 'edge_id'


"""
CONFIGURATIONS
"""
g_sem_units_extractor_config_file = g_src_folder + 'sem_units_ext.conf'
g_lexvec_model_folder = '/home/mf3jh/workspace/lib/lexvec/python/lexvec/'
g_lexvec_vect_file_path = '/home/mf3jh/workspace/lib/lexvec/lexvec.commoncrawl.ngramsubwords.300d.W.pos.bin'
g_phrase_sim_threshold = 0.8
g_phrase_embed_dim = 300
g_datetime_fmt = '%Y%m%d%H%M%S'


def env_check():
    import os
    from os import path
    if not path.exists(g_work_folder):
        raise Exception('g_work_folder does not exist!')
    if not path.exists(g_raw_data_folder):
        raise Exception('g_raw_data_folder does not exist!')
    if not path.exists(g_conf_folder):
        os.mkdir(g_conf_folder)
    if not path.exists(g_raw_txt_info_folder):
        os.mkdir(g_raw_txt_info_folder)
    if not path.exists(g_raw_txt_info_int_folder):
        os.mkdir(g_raw_txt_info_int_folder)
    if not path.exists(g_txt_clean_folder):
        os.mkdir(g_txt_clean_folder)
    if not path.exists(g_txt_clean_int_folder):
        os.mkdir(g_txt_clean_int_folder)
    if not path.exists(g_txt_hash_folder):
        os.mkdir(g_txt_hash_folder)
    if not path.exists(g_txt_hash_int_folder):
        os.mkdir(g_txt_hash_int_folder)
    if not path.exists(g_txt_sent_folder):
        os.mkdir(g_txt_sent_folder)
    if not path.exists(g_txt_sent_int_folder):
        os.mkdir(g_txt_sent_int_folder)
    if not path.exists(g_sem_unit_folder):
        os.mkdir(g_sem_unit_folder)
    if not path.exists(g_sem_unit_int_folder):
        os.mkdir(g_sem_unit_int_folder)
    if not path.exists(g_txt_phrase_folder):
        os.mkdir(g_txt_phrase_folder)
    if not path.exists(g_txt_phrase_int_folder):
        os.mkdir(g_txt_phrase_int_folder)
    if not path.exists(g_phrase_sent_folder):
        os.mkdir(g_phrase_sent_folder)
    if not path.exists(g_phrase_sent_int_folder):
        os.mkdir(g_phrase_sent_int_folder)
    if not path.exists(g_phrase_embed_folder):
        os.mkdir(g_phrase_embed_folder)
    if not path.exists(g_phrase_embed_int_folder):
        os.mkdir(g_phrase_embed_int_folder)
    if not path.exists(g_phrase_cluster_folder):
        os.mkdir(g_phrase_cluster_folder)
    if not path.exists(g_phrase_cluster_int_folder):
        os.mkdir(g_phrase_cluster_int_folder)
    if not path.exists(g_kgraph_folder):
        os.mkdir(g_kgraph_folder)
    if not path.exists(g_kgraph_int_folder):
        os.mkdir(g_kgraph_int_folder)
    if not path.exists(g_pksg_folder):
        os.mkdir(g_pksg_folder)
    if not path.exists(g_pksg_int_folder):
        os.mkdir(g_pksg_int_folder)
    if not path.exists(g_merge_pksg_folder):
        os.mkdir(g_merge_pksg_folder)
    if not path.exists(g_merge_pksg_int_folder):
        os.mkdir(g_merge_pksg_int_folder)
    if not path.exists(g_filter_pksg_folder):
        os.mkdir(g_filter_pksg_folder)
    if not path.exists(g_filter_pksg_int_folder):
        os.mkdir(g_filter_pksg_int_folder)
    if not path.exists(g_t_int_folder):
        os.mkdir(g_t_int_folder)
    if not path.exists(g_pksg_ts_folder):
        os.mkdir(g_pksg_ts_folder)
    if not path.exists(g_data_viz_folder):
        os.mkdir(g_data_viz_folder)
    if not path.exists(g_sent_ts_folder):
        os.mkdir(g_sent_ts_folder)
    if not path.exists(g_sent_ts_int_folder):
        os.mkdir(g_sent_ts_int_folder)
    if not path.exists(g_usr_prj_folder):
        os.mkdir(g_usr_prj_folder)
    if not path.exists(g_usr_prj_int_folder):
        os.mkdir(g_usr_prj_int_folder)


    if not path.exists(g_adj_embed_folder):
        os.mkdir(g_adj_embed_folder)
    if not path.exists(g_curvature_folder):
        os.mkdir(g_curvature_folder)
    if not path.exists(g_curvature_int_folder):
        os.mkdir(g_curvature_int_folder)
    if not path.exists(g_eval_folder):
        os.mkdir(g_eval_folder)
    if not path.exists(g_test_folder):
        os.mkdir(g_test_folder)


env_check()