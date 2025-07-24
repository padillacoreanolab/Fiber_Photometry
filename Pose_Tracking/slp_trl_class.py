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


# --------------------------- utilities --------------------------- #
def fill_missing(Y, kind="linear"):
    """Fill NaNs independently along each time-series column."""
    init_shape = Y.shape
    Y = Y.reshape((init_shape[0], -1))
    for i in range(Y.shape[-1]):
        y = Y[:, i]
        x = np.flatnonzero(~np.isnan(y))
        if x.size == 0:  # all NaN → leave as NaN
            continue
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        # leading/trailing NaNs
        mask = np.isnan(y)
        if mask.any():
            y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
        Y[:, i] = y
    return Y.reshape(init_shape)


def smooth_diff(node_loc: np.ndarray, deriv: int, win=25, poly=3) -> np.ndarray:
    """Savitzky–Golay smoothing/derivative along each coord; if deriv>0, return speed magnitude."""
    out = np.zeros_like(node_loc)
    for c in range(node_loc.shape[1]):
        out[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv)
    if deriv != 0:
        out = np.linalg.norm(out, axis=1)
    return out


# --------------------------- main class --------------------------- #
class SleapTrial(Trial):
    def __init__(self, trial_path, stream_DA, stream_ISOS):
        super().__init__(trial_path, stream_DA, stream_ISOS)
        # raw is (frames, nodes, dims, instances)
        self._raw_locations = None
        self.locations      = None   # (frames, nodes, dims, instances) AFTER crop/fill/smooth
        self.frame_times    = None
        self.in_bout_times  = None
        self.px_to_cm       = None
        self.track_dict     = {}
        self.node_dict      = {}
        self.features_df    = pd.DataFrame()

    # ---------- small vector/angle helpers (ALL angles wrapped then abs) ---------- #
    @staticmethod
    def _wrap180(a_deg):
        """Map degrees to (-180, 180]."""
        return (a_deg + 180.0) % 360.0 - 180.0

    @staticmethod
    def _abs180(a_deg):
        """Unsigned angle in 0..180 (after wrapping)."""
        return np.abs(SleapTrial._wrap180(a_deg))

    def _xy(self, node, track):
        return self.locations[:, self.node_dict[node], :, self.track_dict[track]]  # (F,2)

    def _vec(self, src_node, dst_node, track):
        return self._xy(dst_node, track) - self._xy(src_node, track)               # (F,2)

    @staticmethod
    def _signed_angle(v1, v2):
        """Angle from v1→v2, -180..180."""
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        dot   = (v1 * v2).sum(1)
        return np.degrees(np.arctan2(cross, dot))

    @staticmethod
    def _rho_phi(v):
        """Return (rho, phi_deg) for 2D vectors (F,2); phi is -180..180."""
        rho = np.linalg.norm(v, axis=1)
        phi = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        return rho, phi

    # ---------------------- metadata & DA ---------------------- #
    def keep_only_tracks(self, keep=('subject', 'agent')):
        """
        Keep only the instances named in `keep` (order preserved), drop the rest
        from _raw_locations / locations / track_dict.
        Call this right after load_sleap (and again after filter_sleap_bouts if needed).
        """
        # what tracks exist?
        all_names = list(self.track_dict.keys())

        # pick indices we want (fallback: first two if names missing)
        wanted_names = [n for n in keep if n in self.track_dict]
        if len(wanted_names) < len(keep):
            # names not present; just grab the first len(keep) tracks
            wanted_idx = list(range(min(len(keep), len(all_names))))
            wanted_names = [all_names[i] for i in wanted_idx]
        else:
            wanted_idx = [self.track_dict[n] for n in wanted_names]

        # slice arrays
        self._raw_locations = self._raw_locations[..., wanted_idx]
        if self.locations is not None:
            self.locations = self.locations[..., wanted_idx]

        # rebuild dict
        self.track_dict = {name: i for i, name in enumerate(wanted_names)}

    
    def add_metadata_and_DA(self):
        if self.in_bout_times is None:
            raise RuntimeError("run filter_sleap_bouts() first")
        if getattr(self, 'zscore', None) is None or getattr(self, 'timestamps', None) is None:
            raise RuntimeError("run compute_zscore() first")

        subj = self.subject_name
        first = subj[0].lower() if subj else ''
        if first == 'n':
            region = 'NAc'
        elif first == 'p':
            region = 'mPFC'
        else:
            region = 'unknown'

        df = pd.DataFrame({
            'time_s'        : self.in_bout_times,
            'brain_region'  : region,
            'mouse_identity': subj
        })

        da_ts = self.timestamps
        sleap_ts = self.in_bout_times
        idx = np.searchsorted(da_ts, sleap_ts)
        idx[idx == len(da_ts)] = len(da_ts) - 1
        prev = np.clip(idx - 1, 0, len(da_ts) - 1)
        use_prev = np.abs(da_ts[prev] - sleap_ts) < np.abs(da_ts[idx] - sleap_ts)
        idx[use_prev] = prev[use_prev]
        df['zscore_DA'] = self.zscore[idx]

        bouts = list(self.bouts.items())
        df['intruder_identity'] = [
            next((lbl for lbl, (s, e) in bouts if s <= t <= e), None)
            for t in sleap_ts
        ]

        self.features_df = df
        return df

    # ---------------------- SLEAP I/O & prep ---------------------- #
    def load_sleap(self, h5_path: str, fps: float = 10.0):
        with h5py.File(h5_path, "r") as f:
            raw    = f["tracks"][:].T  # (frames, nodes, dims, instances)
            tracks = [t.decode() for t in f["track_names"][:]]
            nodes  = [n.decode() for n in f["node_names"][:]]

        self._raw_locations = raw
        self.frame_times    = np.arange(raw.shape[0]) / fps
        self.track_dict     = {name: i for i, name in enumerate(tracks)}
        self.node_dict      = {name: i for i, name in enumerate(nodes)}

    def filter_sleap_bouts(self, interp_kind="linear"):
        mask = np.zeros(len(self.frame_times), dtype=bool)
        for start, end in self.bouts.values():
            mask |= (self.frame_times >= start) & (self.frame_times <= end)

        self.in_bout_times = self.frame_times[mask]
        cropped = self._raw_locations[mask, ...]
        self.locations = fill_missing(cropped, kind=interp_kind)

    def smooth_locations(self, win=25, poly=3):
        F, N, D, I = self.locations.shape
        for ni in range(N):
            for ii in range(I):
                traj = self.locations[:, ni, :, ii]
                self.locations[:, ni, :, ii] = smooth_diff(traj, deriv=0, win=win, poly=poly)

    # ---------------------- calibration ---------------------- #
    def calibrate_from_corners(self,
                               corner_h5_path: str,
                               real_width_cm: float,
                               top_left: str = "Top_Left",
                               top_right: str = "Top_Right") -> float:
        with h5py.File(corner_h5_path, "r") as f:
            raw_tracks = f["tracks"][:].T
            node_names = [n.decode() for n in f["node_names"][:]]

        node_idx = {name: i for i, name in enumerate(node_names)}
        if top_left not in node_idx or top_right not in node_idx:
            raise KeyError(f"Corner names must include '{top_left}' and '{top_right}'")

        tl_xy = raw_tracks[:, node_idx[top_left],  :, 0]
        tr_xy = raw_tracks[:, node_idx[top_right], :, 0]

        valid = (~np.isnan(tl_xy).any(axis=1)) & (~np.isnan(tr_xy).any(axis=1))
        if not valid.any():
            raise RuntimeError("No single frame found with both corners labeled!")

        frame_i = np.argmax(valid)
        px_dist = np.linalg.norm(tr_xy[frame_i] - tl_xy[frame_i])
        self.px_to_cm = real_width_cm / px_dist
        return self.px_to_cm

    # ---------------------- feature calculators ---------------------- #
    def node_velocity(self, node: str) -> np.ndarray:
        idx    = self.node_dict[node]
        coords = self.locations[:, idx, :, :]   # (F, 2, I)
        dt     = np.mean(np.diff(self.frame_times))

        speeds = []
        for i in range(coords.shape[-1]):
            xy = coords[..., i]
            dx = np.gradient(xy[:, 0], dt)
            dy = np.gradient(xy[:, 1], dt)
            sp = np.sqrt(dx * dx + dy * dy)
            if self.px_to_cm is not None:
                sp *= self.px_to_cm
            speeds.append(sp)
        return np.stack(speeds, axis=0)

    def subject_velocity(self) -> np.ndarray:
        cen = self.locations.mean(axis=1)       # (F, 2, I)
        dt  = np.mean(np.diff(self.frame_times))
        speeds = []
        for i in range(cen.shape[-1]):
            xy = cen[..., i]
            dx = np.gradient(xy[:, 0], dt)
            dy = np.gradient(xy[:, 1], dt)
            sp = np.sqrt(dx * dx + dy * dy)
            if self.px_to_cm is not None:
                sp *= self.px_to_cm
            speeds.append(sp)
        return np.stack(speeds, axis=0)

    def distance_between(self,
                         node1, track1,
                         node2, track2,
                         normalization_factor=None) -> np.ndarray:
        i1 = self.node_dict[node1]
        i2 = self.node_dict[node2]
        t1 = self.track_dict[track1]
        t2 = self.track_dict[track2]

        c1 = self.locations[:, i1, :, t1]
        c2 = self.locations[:, i2, :, t2]
        d  = np.linalg.norm(c1 - c2, axis=1)
        if self.px_to_cm is not None:
            d *= self.px_to_cm
        return d if normalization_factor is None else d * normalization_factor

    def orientation_between(self, node1, track1, node2, track2):
        """Unsigned angle (0..180) between node1→node2 vector and x-axis."""
        i1 = self.node_dict[node1]
        i2 = self.node_dict[node2]
        t1 = self.track_dict[track1]
        t2 = self.track_dict[track2]
        v = self.locations[:, i2, :, t2] - self.locations[:, i1, :, t1]
        phi = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        return self._abs180(phi)

    # ---------------------- full feature table ---------------------- #
    def compute_pairwise_features(self):
        times = self.in_bout_times

        # ---- node labels ----
        NOSE = 'Nose'
        TAIL = 'Tail_Base'
        EAR_L, EAR_R = 'Left_Ear', 'Right_Ear'
        HEAD_NODE = 'Head'  # if it exists; keep for distances

        # ---------- HEAD VECTORS (ears midpoint → nose) ----------
        ear_mid_res = (self._xy(EAR_L, 'subject') + self._xy(EAR_R, 'subject')) / 2
        ear_mid_int = (self._xy(EAR_L, 'agent')   + self._xy(EAR_R, 'agent'))   / 2
        head_vec_res = self._xy(NOSE, 'subject') - ear_mid_res
        head_vec_int = self._xy(NOSE, 'agent')   - ear_mid_int

        # Body vectors (hind→head) for angles that reference "hind"
        body_vec_res = self._vec(TAIL, NOSE, 'subject')
        body_vec_int = self._vec(TAIL, NOSE, 'agent')

        # ---------- DISTANCES ----------
        dist_head_res__head_int = self.distance_between(HEAD_NODE, 'subject', HEAD_NODE, 'agent')
        dist_head_res__hind_int = self.distance_between(HEAD_NODE, 'subject', TAIL,      'agent')
        dist_head_int__hind_res = self.distance_between(HEAD_NODE, 'agent',   TAIL,      'subject')
        dist_hind_res__hind_int = self.distance_between(TAIL,      'subject', TAIL,      'agent')

        # ---------- ANGLES (wrap to -180..180, then abs -> 0..180) ----------
        ang_head_res__head_int_deg = self._abs180(self._signed_angle(head_vec_res, head_vec_int))
        ang_head_res__hind_int_deg = self._abs180(self._signed_angle(head_vec_res, body_vec_int))
        ang_head_int__hind_res_deg = self._abs180(self._signed_angle(head_vec_int, body_vec_res))

        # ---------- VELOCITIES ----------
        speeds = self.subject_velocity()  # (instances, frames)
        velocity_resident = speeds[self.track_dict['subject']]
        velocity_intruder = speeds[self.track_dict['agent']]

        # ---------- WRITE OUT ----------
        df = self.features_df
        df['time_s'] = times

        df['distance_head_res__head_int']         = dist_head_res__head_int
        df['angle_head_res__head_int_deg']        = ang_head_res__head_int_deg

        df['distance_head_res__hind_int']         = dist_head_res__hind_int
        df['angle_head_res__hind_int_deg']        = ang_head_res__hind_int_deg

        df['distance_head_int__hind_res']         = dist_head_int__hind_res
        df['angle_head_int__hind_res_deg']        = ang_head_int__hind_res_deg

        df['distance_hind_res__hind_int']         = dist_hind_res__hind_int

        df['velocity_resident']                   = velocity_resident
        df['velocity_intruder']                   = velocity_intruder

        return df




    def add_behavior_column(self,
                            df: pd.DataFrame | None = None,
                            time_col: str = "time_s",
                            out_col: str = "behavior_active",
                            mode: str = "all",           # "all" | "first"
                            sep: str = "; "):
        """
        Annotate each row (time point) with the behavior(s) occurring then.

        Parameters
        ----------
        df : DataFrame or None
            DataFrame to tag. If None, uses self.features_df.
        time_col : str
            Column in df containing time stamps (seconds).
        out_col : str
            Name of the new column to write.
        mode : {"all","first"}
            If multiple behaviors overlap a time bin:
            - "all": join their names with `sep`
            - "first": keep only the first one encountered
        sep : str
            Separator when joining multiple behaviors.

        Returns
        -------
        DataFrame
            The same df with a new column `out_col`.
        """
        if df is None:
            if getattr(self, "features_df", None) is None or self.features_df.empty:
                raise RuntimeError("No df provided and self.features_df is empty.")
            df = self.features_df

        # no behavior table → just NA
        if getattr(self, "behaviors", None) is None or self.behaviors.empty:
            df[out_col] = pd.NA
            return df

        times = df[time_col].to_numpy()
        out   = np.full(times.shape, pd.NA, dtype=object)

        beh_df = self.behaviors[["Behavior", "Event_Start", "Event_End"]].to_numpy()
        # loop each behavior interval once
        for beh, start, end in beh_df:
            mask = (times >= start) & (times <= end)
            if not mask.any():
                continue
            if mode == "first":
                # only fill where still NA
                fill_idx = np.where(mask & pd.isna(out))[0]
                out[fill_idx] = beh
            else:  # "all"
                idxs = np.where(mask)[0]
                for i in idxs:
                    if pd.isna(out[i]):
                        out[i] = beh
                    else:
                        out[i] = f"{out[i]}{sep}{beh}"

        df[out_col] = out
        return df
