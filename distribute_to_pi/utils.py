import numpy as np
from datetime import datetime, timedelta


def num_nonzero_elements(arr):
    return len(np.nonzero(arr)[0])

def is_filled(arr):
    return num_nonzero_elements(arr) == len(arr)

def datetime_string(add_seconds=0):
    return (datetime.today() + timedelta(seconds=add_seconds)).strftime('%Y-%m-%d_%H-%M-%S')

def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[n:] - cumsum[:-n]) / float(n)

def audio_volume (recording, smoothing_n=1000):
    volumes = 20 * np.log10(np.abs(recording) + 0.01)
    if smoothing_n > 0:
        volumes = running_mean(volumes, smoothing_n)
    return volumes.flatten()