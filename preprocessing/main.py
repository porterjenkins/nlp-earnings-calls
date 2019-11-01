import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.conference_call_analysis_parallel import loopem
import config.config as cfg


files = os.listdir(cfg.vals["raw_data_dir"])
n_files = len(files)
success_cntr = 0

for cntr, f in enumerate(files):
    fname = cfg.vals["raw_data_dir"] + f
    try:
        prepared_remarks, managers_qa, analysts_qa, data = loopem(fname)
        success_cntr
    except (AttributeError, KeyError) as err:
        pass


    progress = 100 * ((cntr + 1) / n_files)
    print("Progress: {:.4f}".format(progress), end='\r')


print("success: {}/{}".format(success_cntr, n_files))