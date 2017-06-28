# Read labels from .txt file save from tensorboard and copy images in to different folders accordingly.
import os
import glob
import shutil
import json

clustered = []


def read_txt(TXT_PATH):
    """Reads image index from state.txt file saved from tensorboard

    Args:
        TXT_PATH: path to state.txt

    Returns:
        clustered: list of image index

    """
    global clustered
    text_file = (open(TXT_PATH, "r"))
    lines = eval(str(text_file.readlines()))
    lines = lines[0]
    lines = lines[1:-1]
    lines = json.loads(lines)
    #lines = eval(lines)
    clustered = lines.get("selectedPoints")
    return clustered

def read_imgname(DIR_PATH, CLUS_PATH, TXT_PATH):
    """clusters images based on state.txt
    Args:
        DIR_PATH: path to images you want to be clustered  '''
        CLUS_PATH: path to where you save claustered images'''
        TXT_PATH: path to state.txt file. This file is generated from tensorboard'''
    """
    try:
        os.stat(CLUS_PATH)
    except:
        os.mkdir(CLUS_PATH)
    global clustered
    clustered = read_txt(TXT_PATH)
    img_name = glob.glob(DIR_PATH+"*.jpg")
    for i in clustered:
        shutil.copy(img_name[i], CLUS_PATH)
        print img_name[i]


