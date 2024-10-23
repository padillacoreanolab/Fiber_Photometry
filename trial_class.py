import numpy as np
import pandas as pd
import tdt
import matplotlib.pyplot as plt

class Trial:
    def _init_(self, trial_path):
        tdtdata = tdt.read_block(trial_path)
        self.streams = {}
        self.behaviors = {key: value for key, value in tdt_data.epocs.items() if key not in ['Cam1', 'Tick']}

