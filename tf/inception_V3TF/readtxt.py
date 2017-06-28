# Read labels from .txt file save from tensorboard and copy images in to different folders accordingly.
import os
import glob
import shutil
import json

clustered = []


def read_txt(TXT_PATH):
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


