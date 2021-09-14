import logging
import sys
import re

import scenario_settings
from core.sentiment_graphs import build_sgraph_from_json_wrapper


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'build_sgraph_from_json':
        # build_sgraph_from_json 1 0 puretest
        num_task = sys.argv[2]
        job_id = sys.argv[3]
        if re.fullmatch(r'\d+-\d+', job_id):
            l_range = [int(item.strip()) for item in job_id.split('-')]
            start_job_id = l_range[0]
            end_job_id = l_range[1]
            if start_job_id < 0 or end_job_id < 0 or start_job_id > end_job_id:
                raise Exception('[__main__] Invalid job_id: %s' % job_id)
            job_id = [i for i in range(start_job_id, end_job_id + 1)]
        ds_name = sys.argv[4] + '#' + job_id
        build_sgraph_from_json_wrapper(ds_name, num_task, job_id)
    elif cmd == 'test':
        print(sys.argv[2], sys.argv[3], sys.argv[4])
