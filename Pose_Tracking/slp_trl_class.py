import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(PROJECT_ROOT)
from trial_class import Trial
import h5py
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


def smooth_diff(node_loc: np.ndarray, deriv: int, win=25, poly=3) -> np.ndarray:
    # node_loc: (frames, dims)
    out = np.zeros_like(node_loc)
    for c in range(node_loc.shape[1]):
        out[:,c] = savgol_filter(node_loc[:,c], win, poly, deriv)
    if deriv != 0:
        out = np.linalg.norm(out, axis=1)
    return out

class SleapTrial(Trial):
    def __init__(self, trial_path, stream_DA, stream_ISOS):
        super().__init__(trial_path, stream_DA, stream_ISOS)
        # raw is (nodes, dims, instances, frames)
        self._raw_locations  = None  
        # filled & in‐bout also (nodes, dims, instances, frames)
        self.locations       = None  
        self.frame_times     = None  
        self.in_bout_times   = None  
        self.track_dict      = {}
        self.node_dict       = {}
        self.features_df     = pd.DataFrame()


    '''********************************** metadata and DA **********************************'''
    def add_metadata_and_DA(self):
        """
        Augment self.features_df (or create it if empty) by:
          • inferring brain_region from subject_name:
              – if subject_name starts with 'N' or 'n' → 'NAc'
              – if subject_name starts with 'M' or 'm' → 'mPFC'
              – otherwise → 'unknown'
          • mouse_identity = subject_name
          • for each SLEAP frame‐time, find the closest DA timestamp and grab self.zscore there
          • intruder_identity: the bout‐prefix (e.g. 'Short_Term', 'Long_Term', 'Novel') that contains that frame
        """
        # must have already filtered bouts, computed zscore, etc.
        if self.in_bout_times is None:
            raise RuntimeError("run filter_sleap_bouts() first")
        if getattr(self, 'zscore', None) is None or getattr(self, 'timestamps', None) is None:
            raise RuntimeError("run compute_zscore() so self.zscore and self.timestamps exist")

        subj = self.subject_name
        # infer region
        first = subj[0].lower() if subj else ''
        if first == 'n':
            region = 'NAc'
        elif first == 'p':
            region = 'mPFC'

        # build base df
        df = pd.DataFrame({
            'time_s'        : self.in_bout_times,
            'brain_region'  : region,
            'mouse_identity': subj
        })

        # map each sleap‐time to the nearest DA sample
        da_ts = self.timestamps
        sleap_ts = self.in_bout_times
        idx = np.searchsorted(da_ts, sleap_ts)
        idx[idx == len(da_ts)] = len(da_ts) - 1
        prev = np.clip(idx-1, 0, len(da_ts)-1)
        use_prev = np.abs(da_ts[prev] - sleap_ts) < np.abs(da_ts[idx] - sleap_ts)
        idx[use_prev] = prev[use_prev]

        df['zscore_DA'] = self.zscore[idx]

        # intruder_identity by seeing which bout interval each time falls into
        # bout keys look like "Short_Term_1", "Long_Term_2", etc → split off the prefix
        bouts = list(self.bouts.items())
        df['intruder_identity'] = [
            next((lbl for lbl, (s,e) in bouts if s <= t <= e), None)
            for t in sleap_ts
        ]

        # stash it
        self.features_df = df
        return df
    

    '''********************************** SLEAP **********************************'''
    def load_sleap(self, h5_path: str, fps: float = 10.0):
        # 1) Read raw tracks exactly as SLEAP gives them: (nodes, dims, instances, frames)
        with h5py.File(h5_path, "r") as f:
            raw   = f["tracks"][:].T
            tracks = [t.decode() for t in f["track_names"][:]]
            nodes  = [n.decode() for n in f["node_names"][:]]

        self._raw_locations = raw

        # 2) Frame → time axis (1D)
        nF = raw.shape[0]
        self.frame_times   = np.arange(nF) / fps

        # 3) Quick lookup dicts
        self.track_dict = {name:i for i,name in enumerate(tracks)}
        self.node_dict  = {name:i for i,name in enumerate(nodes)}


    def filter_sleap_bouts(self, interp_kind="linear"):
        """
        1) Build a mask over self.frame_times for all bouts
        2) Save in‑bout times in self.in_bout_times
        3) Crop self._raw_locations to only in‑bout frames, then fill_missing
        → self.locations ends up shape (frames, nodes, dims, instances)
        """
        # 1) make mask
        mask = np.zeros(len(self.frame_times), dtype=bool)
        for start, end in self.bouts.values():
            mask |= (self.frame_times >= start) & (self.frame_times <= end)

        # 2) save just the in‑bout timestamps
        self.in_bout_times = self.frame_times[mask]

        # 3) crop raw SLEAP array along the frame axis
        #    self._raw_locations is (frames, nodes, dims, instances)
        cropped = self._raw_locations[mask, ...]  # → (n_bout_frames, nodes, dims, instances)

        # 4) fill any NaNs along each “pixel”‐time‐series
        #    fill_missing expects Y shaped (frames, …) and will flatten axes 1…n
        self.locations = fill_missing(cropped, kind=interp_kind)


    def smooth_locations(self, win=25, poly=3):
        # self.locations: (frames, nodes, dims, instances)
        F,N,D,I = self.locations.shape
        for ni in range(N):
            for ii in range(I):
                traj = self.locations[:, ni, :, ii]  # → (frames, dims)
                sm   = smooth_diff(traj, deriv=0, win=win, poly=poly)
                self.locations[:, ni, :, ii] = sm


    '''********************************** FEATURE CALCULATORS **********************************'''
    def node_velocity(self, node: str) -> np.ndarray:
        """
        Instantaneous speed of a single node for each animal,
        computed as the Euclidean norm of the time‐derivative of x,y.
        
        Returns
        -------
        speeds : ndarray, shape (n_instances, n_frames)
        """
        # grab index & sub‐array: (frames, 2, instances)
        idx    = self.node_dict[node]
        coords = self.locations[:, idx, :, :]  # (F, 2, I)

        # time step
        dt = np.mean(np.diff(self.frame_times))

        # compute d(x,y)/dt by gradient + Euclidean norm
        all_speeds = []
        for i in range(coords.shape[-1]):
            xy = coords[..., i]          # (F, 2)
            dx = np.gradient(xy[:, 0], dt)
            dy = np.gradient(xy[:, 1], dt)
            all_speeds.append(np.sqrt(dx*dx + dy*dy))
        return np.stack(all_speeds, axis=0)  # (I, F)


    def subject_velocity(self) -> np.ndarray:
        """
        Centroid‐based speed for each animal, 
        computed from the average over all nodes.
        
        Returns
        -------
        speeds : ndarray, shape (n_instances, n_frames)
        """
        # first compute per‐frame centroid: (frames, dims, instances)
        centroids = self.locations.mean(axis=1)  # (F, 2, I)

        # time step
        dt = np.mean(np.diff(self.frame_times))

        all_speeds = []
        for i in range(centroids.shape[-1]):
            xy = centroids[..., i]      # (F, 2)
            dx = np.gradient(xy[:, 0], dt)
            dy = np.gradient(xy[:, 1], dt)
            all_speeds.append(np.sqrt(dx*dx + dy*dy))
        return np.stack(all_speeds, axis=0)  # (I, F)


    def distance_between(self,
                         node1, track1,
                         node2, track2,
                         normalization_factor=None):
        i1 = self.node_dict[node1]
        i2 = self.node_dict[node2]
        t1 = self.track_dict[track1]
        t2 = self.track_dict[track2]
        # coords: (frames, dims)
        c1 = self.locations[:, i1, :, t1]
        c2 = self.locations[:, i2, :, t2]
        d  = np.linalg.norm(c1 - c2, axis=1)
        return d if normalization_factor is None else d * normalization_factor
    

    def orientation_between(self,
                            node1, track1,
                            node2, track2):
        i1 = self.node_dict[node1]
        i2 = self.node_dict[node2]
        t1 = self.track_dict[track1]
        t2 = self.track_dict[track2]
        c1 = self.locations[:, i1, :, t1]
        c2 = self.locations[:, i2, :, t2]
        dx = c2[:,0] - c1[:,0]
        dy = c2[:,1] - c1[:,1]
        return np.arctan2(dy, dx)


    '''********************************** FULL FEATURE CSV **********************************'''
    def compute_pairwise_features(self):
        times = self.in_bout_times
        hd   = self.distance_between('Head','subject','Head','agent')
        td   = self.distance_between('Tail_Base','subject','Tail_Base','agent')
        rhid = self.distance_between('Head','subject','Tail_Base','agent')
        ihrd = self.distance_between('Head','agent',  'Tail_Base','subject')
        rhia = self.orientation_between('Head','subject','Tail_Base','agent')
        ihra = self.orientation_between('Head','agent',  'Tail_Base','subject')
        speeds = self.subject_velocity()
        res_v = speeds[self.track_dict['subject']]
        agn_v = speeds[self.track_dict['agent']]

        # ensure our features_df is at least as long as in_bout_times
        # (you could also assert lengths match)
        self.features_df['time_s']                   = times
        self.features_df['head_dist']                = hd
        self.features_df['hind_dist']                = td
        self.features_df['res_head_int_hind_dist']   = rhid
        self.features_df['res_head_int_hind_angle']  = rhia
        self.features_df['int_head_res_hind_dist']   = ihrd
        self.features_df['int_head_res_hind_angle']  = ihra
        self.features_df['resident_velocity']        = res_v
        self.features_df['intruder_velocity']        = agn_v

        return self.features_df



