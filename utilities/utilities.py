#!/usr/bin/env python3
import os
import sys
import argparse #argparse==1.1
from termcolor import colored


def get_all_files(path_to_files, ending='.csv'):
    """This function finds all file in a directory with a given ending

    Args:
        path_to_files: files path
        ending: define ending of the files

    Returns:
        res: a list with all filenames
    """

    dir_path = './'

    filename = ""
    index=[i+1 for i in range(len(path_to_files)) if  path_to_files[i]=='/']

    if len(index)>0:
        dir_path=path_to_files[:index[-1]]
        filename=path_to_files[index[-1]:]

    res = []

    filelist=os.listdir(dir_path)
    filelist.sort()
    #filelist = sorted(filelist, key=int)
    for file in filelist:

        if file.startswith(filename) and file.endswith(ending):
            res.append(f'{dir_path}{file}')
    if len(res) == 0:
        print(f"Warning: No files found at {path_to_files}.")
    return res

def INFO(output):
    try:
        print(colored('[INFO] '+output, 'green'))
    except:
        print(colored('[INFO] '+str(output), 'green'))

def WARN(output):
    try:
        print(colored('[WARNING] '+output, 'yellow'))
    except:
        print(colored('[WARNING] '+str(output), 'yellow'))

def ERROR(output):
    try:
        print(colored('[ERROR] '+output, 'red'))
    except:
        print(colored('[ERROR] '+str(output), 'red'))
    sys.exit()

