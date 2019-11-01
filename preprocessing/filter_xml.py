import os
import sys
import xml.etree.ElementTree as ET
from shutil import copyfile


dir = sys.argv[1]
out_dir = sys.argv[2]
files = os.listdir(dir)
n_files = len(files)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


for cntr, f in enumerate(files):
    fname = dir + "/" + f
    f_split = f.split("_")

    root = ET.parse(fname).getroot()

    if root.attrib['eventTypeId'] == '1' and f_split[1] == 'T.xml':
        out_fname = out_dir + '/' + f
        copyfile(fname, out_fname)



    progress = 100*((cntr + 1) / n_files)
    print("filtering files: {:.4f} done".format(progress), end='\r')