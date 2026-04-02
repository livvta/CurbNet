# -*- coding:utf-8 -*-
# author: CurbNet
# @file: log_util.py 


def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return
