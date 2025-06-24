import os
from rc_extension import Reward_Competition
from trial_class import Trial

class ChannelRTC(Reward_Competition):
    def __init__(self,
                 experiment_folder_path: str,
                 behavior_folder_path: str,
                 keep_channel: str = 'A',
                 subject_filter: str = None):
        """
        keep_channel: 'A' or 'C'  (first vs. second subject in multi-mouse folders)
        subject_filter: e.g. 'nn1', 'pp2'; if not None, only that subject is loaded
        """
        self.keep_channel = keep_channel.upper()
        if self.keep_channel not in ('A','C'):
            raise ValueError("keep_channel must be 'A' or 'C'")
        self.subject_filter = subject_filter
        super().__init__(experiment_folder_path, behavior_folder_path)

    def load_rtc_trials(self):
        """
        Overrides RTC.load_rtc_trials:
        - Single-mouse folders (no underscore) → always load that one (A-channel).
        - Multi-mouse folders (name contains '_') → keep only the A or C mouse,
          *and* only if it matches subject_filter (if provided).
        """
        # clear whatever RTC.__init__ already did
        self.trials.clear()
        self.port_bnc.clear()

        for folder in os.listdir(self.experiment_folder_path):
            folder_path = os.path.join(self.experiment_folder_path, folder)
            if not os.path.isdir(folder_path):
                continue

            parts = folder.split('_')
            if len(parts) > 1:
                # multi-subject folder
                subjA = parts[0]
                # parts[1] is like "nn4-250124-064620"
                subjC = parts[1].split('-',1)[0]
                rest  = parts[1].split('-',1)[1] if '-' in parts[1] else ''

                # decide which subject we WOULD load
                if self.keep_channel == 'A':
                    wanted_subj = subjA
                    trial_key   = f"{subjA}-{rest}" if rest else subjA
                    ch_suffix   = '_465A'
                    iso_suffix  = '_405A'
                    port_val    = 3
                else:
                    wanted_subj = subjC
                    trial_key   = f"{subjC}-{rest}" if rest else subjC
                    ch_suffix   = '_465C'
                    iso_suffix  = '_405C'
                    port_val    = 2

                # filter by subject_filter if given
                if self.subject_filter and self.subject_filter != wanted_subj:
                    continue

                # instantiate exactly one Trial
                self.trials[trial_key] = Trial(folder_path,
                                               ch_suffix,
                                               iso_suffix)
                self.port_bnc[trial_key] = port_val

            else:
                # single-subject folder: folder looks like "nn3-250203-085508"
                subj = folder.split('-',1)[0]
                if self.subject_filter and self.subject_filter != subj:
                    continue
                trial_key = folder
                self.trials[trial_key] = Trial(folder_path,
                                               '_465A',
                                               '_405A')



