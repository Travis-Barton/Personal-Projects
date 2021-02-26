#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:30:06 2020

@author: tbarton
"""
import platform
import pandas as pd
import numpy as np
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import os
import time
import sys
from itertools import islice

# noinspection DuplicatedCode
R = "\033[1;31m"  # RED
G = '\033[1;32m'  # GREEN
Y = "\033[1;33m"  # Yellow
B = "\033[1;34m"  # Blue
N = "\033[0m"  # Reset


def codes_done(title='code complete', msg='', voice=False, speaker='Daniel'):
    os.system("osascript -e 'display notification \"{}\" with title \"{}\"'".format(msg, title))
    if voice and (platform.system() in ['Linux', 'Darwin']):
        os.system(f"say -v {speaker} {title + ',' + msg}")


def delay_print(s, sleep=.1):
    for c in s:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(sleep)


def plot_keras_history(history):
    plt.figure(figsize=[12, 8])
    plt.subplot(2, 1, 1)
    plot(history.history['acc'])
    plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(2, 1, 2)
    plot(history.history['loss'])
    plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.tight_layout()


def window(seq, n=2):
    """
    for the record, this is from stack, not mine ->
    https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator

    Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def island_info(y, trigger_val, stop_ind_inclusive=True):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    # NOTE THIS AGAIN IS NOT MY CODE, BUT SOMETHING I FOUND HERE
    # https://stackoverflow.com/questions/50151417/numpy-find-indices-of-groups-with-same-value

    # Setup "sentients" on either sides to make sure we have setup
    # "ramps" to catch the start and stop for the edge islands
    # (left-most and right-most islands) respectively
    y_ext = np.r_[False, y == trigger_val, False]

    # Get indices of shifts, which represent the start and stop indices
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])

    # Lengths of islands if needed
    lens = idx[1::2] - idx[:-1:2]

    # Using a stepsize of 2 would get us start and stop indices for each island
    return list(zip(idx[:-1:2], idx[1::2] - int(stop_ind_inclusive))), lens


def get_confusion(true, pred):
    if len(true) != len(pred):
        print(f'these values are not the same length \n len pred {len(pred)} ---- len true {len(true)}')
    temp = pd.DataFrame({'true': true, 'pred': pred}).groupby(['true', 'pred']).size().unstack()
    return temp


def most_common(lst):  # from stack: https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
    return max(set(lst), key=lst.count)


def gaussian_mle(data):
    # Code was found at:
    # https://stackoverflow.com/questions/51342724/how-to-estimate-gaussian-distribution-parameters-using-mle-in-python
    # used for finding parameters of 2d gaussian dist data
    mu = data.mean(axis=0)
    var = (data-mu).T @ (data-mu) / data.shape[0]  # this is slightly suboptimal, but instructive

    return mu, var
