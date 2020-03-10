# Original script downloaded from:
# https://github.com/facebookresearch/video-nonlocal-net/blob/master/process_data/kinetics/downscale_video_joblib.py

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import fnmatch
import glob
import json
import os
import shutil
import subprocess
import uuid

from joblib import delayed
from joblib import Parallel
import pandas as pd

def downscale_clip(inname, outname):

    status = False
    inname = '"%s"' % inname
    outname = '"%s"' % outname
    command = "ffmpeg  -loglevel panic -i {} -filter:v scale=\"trunc(oh*a/2)*2:256\" -q:v 1 -c:a copy {}".format( inname, outname)
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return status, err.output

    status = os.path.exists(outname)
    return status, 'Downscaled'


def downscale_clip_wrapper(row, folder_path, output_path):

    nameset  = row.split('/')
    videoname = nameset[-1]
    classname = nameset[-2]

    output_folder = output_path + classname
    if os.path.isdir(output_folder) is False:
        try:
            os.mkdir(output_folder)
        except:
            print(output_folder)

    inname = folder_path + classname + '/' + videoname
    outname = output_path + classname + '/' + videoname

    downscaled, log = downscale_clip(inname, outname)
    return downscaled


def downscale_video_joblib(file_src,
                           folder_path,
                           output_path):
    file_list = []

    f = open(file_src, 'r')

    for line in f:
        rows = line.split()
        fname = rows[0]
        file_list.append(fname)

    f.close()

    status_lst = Parallel(n_jobs=16)(delayed(downscale_clip_wrapper)(row, folder_path, output_path) for row in file_list)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_src', type=str, default='trainlist.txt')
    parser.add_argument('--folder_path', type=str, default='YOUR_DATASET_FOLDER/train/')
    parser.add_argument('--output_path', type=str, default='YOUR_DATASET_FOLDER/train_256/')

    return parser.parse_args()

def main():
    args = parse_args()

    downscale_video_joblib(args.file_src,
                           args.folder_path,
                           args.output_path)

if __name__ == '__main__':
    main()
