import logging
import sys
import re

from core.txt_cleansing import txt_clean_wrapper
from snt_ana_frcst_txt_clean import gen_txt_clean_tasks


def rm_weird_nn(input_txt):
    return re.sub(r'([^a-zA-Z,^])?nn', r'\1 ', input_txt)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        # gen_tasks 10 0 ca tw_raw_txt
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        raw_txt_col = sys.argv[5]
        gen_txt_clean_tasks(ds_name, num_task, job_id, raw_txt_col)
    elif cmd == 'txt_clean':
        # txt_clean 10 0 ca
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        txt_clean_wrapper(ds_name, num_task, job_id, extra_clean_func=rm_weird_nn)
