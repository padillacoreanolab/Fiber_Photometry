import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(PROJECT_ROOT)

import cv2
import numpy as np
import matplotlib.pyplot as plt
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
        """
        Load the .analysis.h5 exactly the same way as your overlay does:
          tracks has shape (inst, dims, nodes, frames) → transpose to
          (frames, nodes, dims, inst).
        """
        with h5py.File(h5_path, "r") as f:
            raw = f["tracks"][:]                             # (inst, dims, nodes, frames)
            # reorder to (frames, nodes, dims, instances)
            tracks = raw.transpose((3, 2, 1, 0))             # (F, N, D, M)

            track_names = [t.decode() for t in f["track_names"][:]]
            node_names  = [n.decode() for n in f["node_names"][:]]

        # now store in the exact same shape your overlay expects
        self._raw_locations = tracks
        self.frame_times    = np.arange(tracks.shape[0]) / fps

        # build lookup dicts
        self.track_dict     = {name: i for i, name in enumerate(track_names)}
        self.node_dict      = {name: i for i, name in enumerate(node_names)}


    def filter_sleap_bouts(self, interp_kind="linear"):
        """
        Builds two views on self._raw_locations:
        • self.locations_full    : (F, N, D, M), NaN when outside bouts
        • self.locations_cropped : (n_bout, N, D, M), only the bout frames

        Also stores:
        • self.frame_indices     : length‐n_bout array of original frame numbers
        • self.in_bout_times     : length‐n_bout array of corresponding times
        """
        # 1) make a boolean mask over all original frames
        mask = np.zeros(len(self.frame_times), dtype=bool)
        for start, end in self.bouts.values():
            mask |= (self.frame_times >= start) & (self.frame_times <= end)

        # 2) record which frame‐indices survive
        self.frame_indices = np.flatnonzero(mask)  # e.g. [102,103,104,…]

        # 3) slice out only those frames and interpolate gaps
        cropped = self._raw_locations[self.frame_indices, ...]       # (n_bout, N, D, M)
        filled  = fill_missing(cropped, kind=interp_kind)

        # 4) store the cropped view
        self.locations_cropped = filled
        self.in_bout_times     = self.frame_times[self.frame_indices]

        # 5) build a full‐length version with NaNs outside your bouts
        full_shape  = self._raw_locations.shape  # (F, N, D, M)
        full_masked  = np.full(full_shape, np.nan, dtype=filled.dtype)
        full_masked[self.frame_indices, ...] = filled
        self.locations_full = full_masked

        # 6) optionally choose your “default” locations array
        #    (you can switch this back and forth in your pipeline)
        # self.locations = self.locations_full
        # or
        # self.locations = self.locations_cropped



    def smooth_locations(self, win: int = 25, poly: int = 3):
        """
        Smooth the SLEAP tracks with a Savitzky–Golay filter.

        Operates on self.locations_cropped (only bout frames), then
        rebuilds self.locations_full so the smoothed data is in both views.

        Must call filter_sleap_bouts(...) before this.
        """
        # 1) sanity check
        if not hasattr(self, "locations_cropped"):
            raise RuntimeError("Call filter_sleap_bouts() before smooth_locations()")

        # 2) smooth the cropped trajectories
        cropped = self.locations_cropped    # shape = (n_bout, N, D, M)
        Fc, N, D, M = cropped.shape
        smoothed = np.zeros_like(cropped)
        for ni in range(N):
            for mi in range(M):
                traj = cropped[:, ni, :, mi]         # (Fc, 2)
                smoothed[:, ni, :, mi] = smooth_diff(traj, deriv=0, win=win, poly=poly)

        # 3) overwrite the cropped view
        self.locations_cropped = smoothed

        # 4) rebuild the full‐length, NaN‐masked view
        full_shape = self._raw_locations.shape  # (F, N, D, M)
        full = np.full(full_shape, np.nan, dtype=smoothed.dtype)
        full[self.frame_indices, ...] = smoothed
        self.locations_full = full

        # 5) keep self.locations pointing to whichever you prefer;
        #    here we default to the cropped (you could swap to full instead)
        self.locations = self.locations_cropped

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


    # Proximity functions
    def compute_social_ellipse_masks(self,
                                 a_cm: float,
                                 b_cm: float,
                                 angle_src: str   = 'Tail_Base',
                                 angle_dst: str   = 'Nose',
                                 use_full: bool   = True
                                 ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute two full‐length boolean masks over *_raw_locations* (shape F×...):
        - mask_agent_in_subject: True when agent’s nose inside subject’s ellipse
        - mask_subject_in_agent: True when subject’s nose inside agent’s ellipse

        If use_full=True, uses self.locations_full (F×N×2×M).
        """
        # pick the array that covers every raw frame
        loc = (self.locations_full if use_full and hasattr(self, 'locations_full')
            else self._raw_locations)   # (F, N_nodes, 2, M_instances)

        F, N, D, M = loc.shape

        def _one_mask(center_track, target_track):
            t_c = self.track_dict[center_track]
            t_t = self.track_dict[target_track]

            # —(1) ellipse center = per‐frame mean over *all* nodes of the center animal
            #    (that approximates its body center / COM)
            C = np.nanmean(loc[:, :, :, t_c], axis=1)   # shape (F,2)

            # —(2) orientation vector on the same animal
            src = self.node_dict[angle_src]
            dst = self.node_dict[angle_dst]
            V = loc[:, dst, :, t_c] - loc[:, src, :, t_c]  # (F,2)
            thetas = np.arctan2(V[:,1], V[:,0])

            # —(3) nose of the target animal
            idx_nose = self.node_dict['Nose']
            P = loc[:, idx_nose, :, t_t]                   # (F,2)

            # —(4) convert cm→px
            if self.px_to_cm:
                a_px, b_px = a_cm / self.px_to_cm, b_cm / self.px_to_cm
            else:
                a_px, b_px = a_cm, b_cm

            # —(5) rotate into ellipse frame & test
            d = P - C                                      # (F,2)
            cos_t, sin_t = np.cos(thetas), np.sin(thetas)
            x_rot =  cos_t * d[:,0] + sin_t * d[:,1]
            y_rot = -sin_t * d[:,0] + cos_t * d[:,1]
            return (x_rot**2 / a_px**2 + y_rot**2 / b_px**2) <= 1

        self.mask_agent_in_subject = _one_mask('subject', 'agent')
        self.mask_subject_in_agent = _one_mask('agent',   'subject')
        return self.mask_agent_in_subject, self.mask_subject_in_agent



    def add_social_labels(self, a_cm: float, b_cm: float):
        """
        Adds two columns to self.features_df, tagging each row (bout‐frame)
        whether agent→subject or subject→agent sociality was True.
        """
        # 1) make sure we have full‐length locations
        if not hasattr(self, 'locations_full'):
            self.filter_sleap_bouts()

        # 2) compute them (length = n_raw_frames)
        maskA, maskS = self.compute_social_ellipse_masks(a_cm, b_cm)

        # 3) map your cropped‐indices → raw frames
        raw_idxs = self.frame_indices  # len = n_bout_frames

        # 4) sanity check
        if raw_idxs.max() >= len(maskA):
            raise IndexError(f"Frame index {raw_idxs.max()} ≥ mask length {len(maskA)}")

        # 5) slice
        a_flags = maskA[raw_idxs]
        s_flags = maskS[raw_idxs]

        # 6) write‐into your df
        self.features_df['agent_in_subject'] = np.where(a_flags, "Yes", "No")
        self.features_df['subject_in_agent'] = np.where(s_flags, "Yes", "No")
        return self.features_df
        

    def load_video(self, video_path: str):
        """Open the arena video so we can grab raw frames later."""
        self._video = cv2.VideoCapture(video_path)
        if not self._video.isOpened():
            raise IOError(f"Cannot open video {video_path}")

    def get_frame(self, frame_i: int) -> np.ndarray:
        """
        Seek to frame_i in the *filtered* timebase and return the correct
        original‐video frame (H×W×3, RGB).
        """
        # Map from your filtered index → raw video index
        if hasattr(self, "frame_indices"):
            orig_frame = int(self.frame_indices[frame_i])
        else:
            orig_frame = frame_i

        self._video.set(cv2.CAP_PROP_POS_FRAMES, orig_frame)
        ok, bgr = self._video.read()
        if not ok:
            raise IndexError(f"Frame {orig_frame} could not be read")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    


    # Plotting
    def plot_raw_frame_skeleton(self, frame_i: int, video_path: str = None):
        """
        Plot video‐frame #frame_i with the *raw* (uncropped, unsmoothed) SLEAP skeleton
        for both 'subject' and 'agent'.  If *any* node at that frame is NaN, raises.

        Parameters
        ----------
        frame_i : int
            The *raw* video frame to grab (0..n_frames-1).
        video_path : str, optional
            If self._video isn’t yet open, will load this path.
        """
        # 1) bounds check
        raw = self._raw_locations  # (frames, nodes, dims, instances)
        F = raw.shape[0]
        if not (0 <= frame_i < F):
            raise IndexError(f"Frame {frame_i} out of range [0, {F})")

        # 2) NaN check
        this_frame = raw[frame_i]  # (nodes, dims, instances)
        if np.isnan(this_frame).any():
            raise ValueError(f"SLEAP data contains NaN at raw frame {frame_i}")

        # 3) ensure video loaded
        if not hasattr(self, "_video") or self._video is None:
            if video_path is None:
                raise ValueError("No video loaded—pass video_path")
            self.load_video(video_path)

        # 4) grab exact video frame
        self._video.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        ok, bgr = self._video.read()
        if not ok:
            raise RuntimeError(f"Could not read video frame {frame_i}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 5) plot
        fig, ax = plt.subplots(figsize=(8,6))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Raw SLEAP skeleton — frame {frame_i}")

        colors = {"subject":"lime", "agent":"red"}
        for track_name, t_idx in self.track_dict.items():
            if track_name not in colors:
                continue
            pts = this_frame[:, :, t_idx]  # (nodes, dims)
            xs, ys = pts[:,0], pts[:,1]    # x=col, y=row
            ax.scatter(xs, ys,
                       c=colors[track_name],
                       s=50, edgecolor="k",
                       label=track_name)
            ax.plot(xs, ys,
                    c=colors[track_name],
                    lw=2, alpha=0.7)

        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
