import numpy as np
import pandas as pd
import tdt
import matplotlib.pyplot as plt
import os

class Trial:
    def __init__(self, trial_path):
        tdtdata = tdt.read_block(trial_path)
        self.streams = {}
        self.behaviors = {key: value for key, value in tdtdata.epocs.items() if key not in ['Cam1', 'Tick']}
        
        self.subject_name = os.path.basename(trial_path).split('-')[0]

        self.fs = tdtdata.streams["_465A"].fs
        self.timestamps = np.arange(len(tdtdata.streams['_465A'].data)) / self.fs


