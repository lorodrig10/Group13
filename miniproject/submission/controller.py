import numpy as np
from miniproject.simulation import MiniprojectSimulation
import matplotlib.pyplot as plt
from spike_bin_from_vision import spike_bin_masks_from_raw_vision

SHOW_PRINTS = True
print_frequency = 1000


class Controller:
    """Vision + olfaction hybrid controller for COBAR miniproject levels 0-3.

    Olfaction follows the week 4 solutions notebook (weighted attractive / aversive
    gradient climbing). Level 2+ grass spikes use ``spike_bin`` masks; level 3 adds a
    light heading-based wind bias using ``world.flow_velocity``.
    """

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self.show_prints = False

        self._terrain = bool(sim.enable_terrain)
        self._grass = bool(sim.enable_grass)
        self._wind = bool(sim.enable_wind)

        self.odor_smooth = None
        # Faster smoothing on hills so gradient estimates track better.
        self.alpha = 0.0025 if self._terrain else 0.001

        self.odor_base = 1.25 if self._terrain else 1.05
        # Week-4 notebook used tanh(bias**2)*sign(bias), which kills weak gradients (bias**2 ~ 0).
        # Use tanh(scale*bias) so small left/right odor differences still steer.
        self.odor_bias_tanh_scale = 3.2 if self._terrain else 2.4
        self.odor_modulation_gain = 0.92 if self._terrain else 0.84
        # When grass + vision blending is active, amplify odor left/right contrast so odor isn't drowned out.
        self.odor_side_boost_with_grass = 1.38 if self._terrain else 1.28
        # If the fly consistently turns the wrong way toward the banana, set True (sensor vs reshape order).
        self.odor_left_right_flip = False

        # --- Spike obstacle blending (pixels -> mix weight) ---
        # Slightly higher thresholds so odor + threading dominates in dense spikes.
        self.spike_blend_t0_px = 58.0
        self.spike_blend_t1_px = 98.0
        self.spike_blend_smooth_alpha = 0.12
        self._spike_blend_w = 0.0

        self.avoid_base_speed = 1.12
        self.avoid_steer_gain = 0.88
        self.avoid_steer_k = 2.75
        self.avoid_centroid_gain = 0.52
        # When L/R spike heights match (corridor), scale avoidance blend down toward odor.
        self.avoid_symmetry_w_min = 0.30
        self.avoid_symmetry_w_span = 0.70
        # Clear binocular forward strips -> trust odor / go straight more.
        self.avoid_gap_w_reduce = 0.52
        self.avoid_steer_deadband = 0.11
        # Frontal wall (both eyes see spikes ahead): recover avoidance + pivot-style turn.
        self.frontal_gap_clear_ref = 0.44
        self.frontal_pair_px_floor = 32.0
        self.frontal_pair_px_scale = 55.0
        self.frontal_sym_relief = 0.62
        self.pivot_drive_slow = 0.18
        self.pivot_drive_fast = 1.52
        self.pivot_mix_gain = 0.82
        # Imminent contact: tall vertical spike run (px): floor avoidance blend, relax gap logic.
        self.emergency_px_start = 76.0
        self.emergency_px_span = 58.0
        self.emergency_w_floor_max = 0.92
        self.emergency_gap_scale_floor = 0.62
        self.emergency_brake_strength = 0.42
        # Both eyes see spikes (pair_px): extra urgency even if max run is only ~70-100px.
        self.emergency_dual_pair_mid = 48.0
        self.emergency_dual_mx_mid = 62.0

        # Wind: steer opposite lateral push from unit wind direction (level 3+).
        self.wind_steer_gain = 0.38
        self.wind_mix = 0.42

        self._drives_lp = None
        self.drives_lp_beta = 0.26 if self._terrain else 0.18
        self._last_emergency = 0.0

    def step(self, sim: MiniprojectSimulation, step):
        self.show_prints = SHOW_PRINTS and step % print_frequency == 0

        raw_olfaction = sim.get_olfaction(sim.fly.name)

        if self.odor_smooth is None:
            self.odor_smooth = raw_olfaction.copy()
        else:
            self.odor_smooth = (
                (1 - self.alpha) * self.odor_smooth
                + self.alpha * raw_olfaction
            )

        odor_drives = self._odor_drives_from_smooth()
        if self._grass:
            mu = 0.5 * float(odor_drives[0] + odor_drives[1])
            delta = odor_drives - mu
            odor_drives = np.clip(
                mu + delta * float(self.odor_side_boost_with_grass), 0.18, None
            )

        feat = None
        self._last_emergency = 0.0
        if self._grass:
            feat = self._spike_features(sim)
            avoid_drives = self._avoid_from_features(feat)
            w = feat["blend_w"]
            drives = (1.0 - w) * odor_drives + w * avoid_drives
            drives = self._apply_frontal_pivot(drives, odor_drives, avoid_drives, feat)
            drives = self._apply_collision_brake(drives, feat)
            self._last_emergency = float(feat.get("emergency", 0.0))
        else:
            w = 0.0
            drives = odor_drives.copy()

        drives = self._apply_wind_bias(sim, drives)
        drives = self._lowpass_drives(drives)

        if self.show_prints:
            if self._grass and feat is not None:
                print(
                    f"Spike threat px: L={feat['left_px']:.0f}, R={feat['right_px']:.0f} | "
                    f"w_mix={w:.3f} w_raw={feat.get('blend_w_raw', w):.3f} | "
                    f"gap={feat.get('gap_clear', 0.0):.2f} sym={feat.get('symmetry_scale', 0.0):.2f} "
                    f"front={feat.get('frontal_pressure', 0.0):.2f} "
                    f"emg={feat.get('emergency', 0.0):.2f}"
                )
            else:
                print(f"(no grass) odor-only | wind_level={self._wind}")

        joint_angles, adhesion = self.turning_controller.step(drives)

        if step > 0 and self.show_prints:
            fly_vision = np.concatenate(sim.get_raw_vision(sim.fly.name), axis=-2)
            plt.figure(figsize=(10, 4))
            plt.imshow(fly_vision)
            plt.title(f"Vision step {step} - blended (w={w:.2f})")
            plt.axis("off")
            plt.show()

        return joint_angles, adhesion

    def _odor_drives_from_smooth(self):
        """Week 4 ``solutions_olfaction.ipynb``: weighted attractive vs aversive bias.

        Miniproject banana worlds expose a single odor dimension (shape (4, 1));
        course arenas may use two dimensions (attractive + aversive).
        """
        x = np.asarray(self.odor_smooth, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        attractive_gain = -520.0
        aversive_gain = 82.0

        attractive_intensities = np.average(
            x[:, 0].reshape(2, 2), axis=0, weights=[9.0, 1.0]
        )

        amean = attractive_intensities.mean()
        asym_attr = attractive_intensities[0] - attractive_intensities[1]
        if self.odor_left_right_flip:
            asym_attr = -asym_attr
        attractive_bias = (
            attractive_gain * asym_attr / amean if abs(amean) > 1e-12 else 0.0
        )

        if x.shape[1] >= 2:
            aversive_intensities = np.average(
                x[:, 1].reshape(2, 2), axis=0, weights=[10.0, 0.0]
            )
            vmean = aversive_intensities.mean()
            asym_av = aversive_intensities[0] - aversive_intensities[1]
            if self.odor_left_right_flip:
                asym_av = -asym_av
            aversive_bias = (
                aversive_gain * asym_av / vmean if abs(vmean) > 1e-12 else 0.0
            )
        else:
            aversive_bias = 0.0

        effective_bias = aversive_bias + attractive_bias
        sc = float(self.odor_bias_tanh_scale)
        effective_bias_norm = float(np.tanh(np.clip(sc * effective_bias, -5.0, 5.0)))

        control_signal = np.ones(2, dtype=float) * self.odor_base
        side_to_modulate = int(effective_bias_norm > 0)
        modulation_amount = np.abs(effective_bias_norm) * float(self.odor_modulation_gain)
        control_signal[side_to_modulate] -= modulation_amount

        return np.clip(control_signal, 0.18, None)

    def _thorax_heading_xy(self, sim: MiniprojectSimulation) -> float:
        segments = sim.fly.get_bodysegs_order()
        thorax_idx = next(
            (i for i, seg in enumerate(segments) if seg.name == "c_thorax"),
            0,
        )
        q = sim.get_body_rotations(sim.fly.name)[thorax_idx].astype(float)
        w, qx, qy, qz = q[0], q[1], q[2], q[3]
        return float(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))

    def _apply_wind_bias(self, sim: MiniprojectSimulation, drives: np.ndarray) -> np.ndarray:
        if not self._wind:
            return drives

        fv = np.asarray(sim.world.flow_velocity[:2], dtype=float)
        norm = float(np.linalg.norm(fv))
        if norm < 1e-9:
            return drives

        heading = self._thorax_heading_xy(sim)
        forward = np.array([np.cos(heading), np.sin(heading)], dtype=float)
        w_hat = fv / norm
        cross = float(w_hat[0] * forward[1] - w_hat[1] * forward[0])

        steer_w = float(np.clip(self.wind_steer_gain * cross, -1.0, 1.0))
        base = 0.5 * float(drives[0] + drives[1])
        g = float(self.avoid_steer_gain)
        wind_l = base * (1.0 + g * steer_w)
        wind_r = base * (1.0 - g * steer_w)
        wind_drives = np.array([wind_l, wind_r], dtype=float)

        m = float(self.wind_mix)
        out = (1.0 - m) * drives + m * wind_drives
        return np.clip(out, 0.18, None)

    def _lowpass_drives(self, drives: np.ndarray) -> np.ndarray:
        emg = float(np.clip(self._last_emergency, 0.0, 1.0))
        b = float(np.clip(self.drives_lp_beta * (1.0 + 1.45 * emg), 0.0, 0.92))
        if self._drives_lp is None:
            self._drives_lp = drives.copy()
        else:
            self._drives_lp = (1.0 - b) * self._drives_lp + b * drives
        return np.clip(self._drives_lp.copy(), 0.18, None)

    def _forward_binocular_gap_clearance(
        self, left_m: np.ndarray, right_m: np.ndarray
    ) -> float:
        """How clear the forward/nasal strips are (1 = open gap, 0 = blocked).

        Left-eye forward region is the nasal (right-hand) columns; right-eye forward is
        the temporal (left-hand) columns. When both are relatively empty, a corridor exists.
        """
        bl = (np.asarray(left_m) > 0).astype(np.float64)
        br = (np.asarray(right_m) > 0).astype(np.float64)
        _hl, wl = bl.shape
        _hr, wr = br.shape
        frac = 0.44
        i0 = max(0, int(round(wl * (1.0 - frac))))
        strip_l = bl[:, i0:]
        j1 = max(1, int(round(wr * frac)))
        strip_r = br[:, :j1]
        occ = 0.5 * (float(np.mean(strip_l)) + float(np.mean(strip_r)))
        # Compress occupancy slightly so partial gaps still read as clear enough.
        return float(np.clip(1.0 - occ * 1.12, 0.0, 1.0))

    def _forward_binocular_strip_occ(self, left_m: np.ndarray, right_m: np.ndarray) -> float:
        """Mean occupancy [0,1] in forward binocular strips (same ROI as gap_clear)."""
        bl = (np.asarray(left_m) > 0).astype(np.float64)
        br = (np.asarray(right_m) > 0).astype(np.float64)
        _hl, wl = bl.shape
        _hr, wr = br.shape
        frac = 0.44
        i0 = max(0, int(round(wl * (1.0 - frac))))
        strip_l = bl[:, i0:]
        j1 = max(1, int(round(wr * frac)))
        strip_r = br[:, :j1]
        return float(0.5 * (np.mean(strip_l) + np.mean(strip_r)))

    def _spike_horizontal_bias(self, spikes_bin_u8: np.ndarray) -> float:
        """[-1, 1]: lateral mass of spikes within the eye ROI (nasal side -> +)."""
        m = np.asarray(spikes_bin_u8) > 0
        _h, w = m.shape
        xs = np.flatnonzero(m.any(axis=0))
        if xs.size == 0:
            return 0.0
        cols = m[:, xs]
        weights = cols.sum(axis=0).astype(float)
        sw = weights.sum()
        if sw <= 0:
            return 0.0
        cx = float(np.dot(xs.astype(float), weights) / sw)
        return float((cx / max(w - 1, 1) - 0.5) * 2.0)

    def _spike_features(self, sim: MiniprojectSimulation) -> dict:
        raw_vision = sim.get_raw_vision(sim.fly.name)
        masks = spike_bin_masks_from_raw_vision(raw_vision, layout="two_frames")

        left_stack_h, left_cols = self._max_white_stack_height_and_cols_from_bottom(
            masks["left"]
        )
        right_stack_h, right_cols = self._max_white_stack_height_and_cols_from_bottom(
            masks["right"]
        )
        left_px = float(left_stack_h)
        right_px = float(right_stack_h)

        left_bias = self._spike_horizontal_bias(masks["left"])
        right_bias = self._spike_horizontal_bias(masks["right"])

        if self.show_prints:
            print(
                "Spike max vertical white run (px per col): "
                f"L={int(left_stack_h)} | R={int(right_stack_h)}"
            )
            plt.figure(figsize=(10, 4))
            plt.suptitle("spikes_bin (raw mask) - left / right")
            ax1 = plt.subplot(1, 2, 1)
            ax1.set_title(f"Left spikes_bin - max vertical run={left_stack_h}px")
            ax1.imshow(
                self._stack_columns_overlay_rgb(masks["left"], left_stack_h, left_cols),
                interpolation="nearest",
            )
            ax1.axis("off")

            ax2 = plt.subplot(1, 2, 2)
            ax2.set_title(f"Right spikes_bin - max vertical run={right_stack_h}px")
            ax2.imshow(
                self._stack_columns_overlay_rgb(masks["right"], right_stack_h, right_cols),
                interpolation="nearest",
            )
            ax2.axis("off")
            plt.show()

        urgent = max(left_px, right_px)
        t0 = float(self.spike_blend_t0_px)
        t1 = float(self.spike_blend_t1_px)
        if t1 <= t0:
            w_new = 1.0 if urgent > t0 else 0.0
        else:
            w_new = float(np.clip((urgent - t0) / (t1 - t0), 0.0, 1.0))
        a = float(np.clip(self.spike_blend_smooth_alpha, 0.0, 1.0))
        self._spike_blend_w = (1.0 - a) * float(self._spike_blend_w) + a * w_new
        blend_w = float(np.clip(self._spike_blend_w, 0.0, 1.0))

        # Tight spike fields: similar left/right threat often means a corridor/gap, not
        # turn away from one side heuristic. Down-weight avoidance so the fly threads gaps.
        mx = max(left_px, right_px, 1e-6)
        symmetry_scale = float(np.clip(abs(left_px - right_px) / mx, 0.0, 1.0))
        corridor_scale = self.avoid_symmetry_w_min + self.avoid_symmetry_w_span * symmetry_scale

        gap_clear = self._forward_binocular_gap_clearance(masks["left"], masks["right"])
        gap_scale = 1.0 - float(self.avoid_gap_w_reduce) * gap_clear

        pair_px = min(left_px, right_px)
        # fused threat: closeness on BOTH sides matters for head-on gaps / tunnels
        fused_px = float(0.52 * mx + 0.48 * pair_px)

        strip_occ = self._forward_binocular_strip_occ(masks["left"], masks["right"])
        occ_emergency = float(np.clip((strip_occ - 0.12) / 0.44, 0.0, 1.0))

        mask_h = float(masks["left"].shape[0])
        rel_urgency = float(fused_px / max(mask_h, 1.0))
        emergency = float(
            np.clip(
                (fused_px - float(self.emergency_px_start)) / max(self.emergency_px_span, 1e-6),
                0.0,
                1.0,
            )
        )
        if rel_urgency >= 0.34:
            emergency = max(emergency, float(np.clip((rel_urgency - 0.34) / 0.44, 0.0, 1.0)))

        emergency = max(emergency, occ_emergency * 0.98)

        dual_tunnel = float(
            np.clip((pair_px - self.emergency_dual_pair_mid) / 42.0, 0.0, 1.0)
            * np.clip((mx - self.emergency_dual_mx_mid) / 48.0, 0.0, 1.0)
        )
        emergency = max(emergency, dual_tunnel * 0.88)

        if emergency > 0.04:
            gap_scale = max(gap_scale, float(self.emergency_gap_scale_floor + (1.0 - self.emergency_gap_scale_floor) * emergency))

        frontal_close = float(
            np.clip(
                (self.frontal_gap_clear_ref - gap_clear) / max(self.frontal_gap_clear_ref, 1e-6),
                0.0,
                1.0,
            )
        )
        sym_relief = (
            float(
                np.clip(
                    (pair_px - self.frontal_pair_px_floor)
                    / max(self.frontal_pair_px_scale, 1e-6),
                    0.0,
                    1.0,
                )
            )
            * frontal_close
        )
        corridor_scale_eff = corridor_scale + (1.0 - corridor_scale) * (
            float(self.frontal_sym_relief) * sym_relief
        )

        blend_w_mix = float(np.clip(blend_w * corridor_scale_eff * gap_scale, 0.0, 1.0))

        if emergency > 0.035:
            w_floor = float(
                0.41
                + (float(self.emergency_w_floor_max) - 0.41) * np.power(emergency, 0.82)
            )
            blend_w_mix = float(max(blend_w_mix, w_floor))

        frontal_pressure = frontal_close * sym_relief * float(
            np.clip(mx / 95.0, 0.0, 1.0)
        )
        frontal_pressure = float(max(frontal_pressure, emergency * 0.92))

        return {
            "left_px": left_px,
            "right_px": right_px,
            "left_bias": left_bias,
            "right_bias": right_bias,
            "blend_w": blend_w_mix,
            "blend_w_raw": blend_w,
            "gap_clear": gap_clear,
            "symmetry_scale": symmetry_scale,
            "frontal_pressure": frontal_pressure,
            "pair_px": float(pair_px),
            "fused_px": fused_px,
            "strip_occ": strip_occ,
            "mask_h": mask_h,
            "rel_urgency": rel_urgency,
            "emergency": float(np.clip(emergency, 0.0, 1.0)),
        }

    def _avoid_from_features(self, feat: dict) -> np.ndarray:
        h_l = feat["left_px"]
        h_r = feat["right_px"]
        hb_l = feat["left_bias"]
        hb_r = feat["right_bias"]

        if self.show_prints:
            print(
                f"AVOID features: h L={h_l:.1f} R={h_r:.1f} | "
                f"lat L={hb_l:.3f} R={hb_r:.3f}"
            )

        # Normalize heights using a soft scale (~typical tall spike run in px).
        scale = 110.0
        l = h_l / scale
        r = h_r / scale
        height_term = l - r

        # Left-eye nasal congestion vs right-eye pushes steer positive (turn right).
        centroid_term = self.avoid_centroid_gain * (hb_l - hb_r)

        combined = float(height_term + centroid_term)
        db = float(self.avoid_steer_deadband)
        if abs(combined) < db:
            combined *= 0.22

        # If forward corridor looks open, damp sideways avoidance so we don't weave.
        gc = float(feat.get("gap_clear", 0.0))
        emg = float(feat.get("emergency", 0.0))
        gc_mul = 1.0 - 0.55 * gc * (1.0 - 0.88 * emg)
        combined *= gc_mul

        fp = float(feat.get("frontal_pressure", 0.0))
        if fp > 0.08:
            combined *= 1.0 + 1.55 * fp

        steer_k = float(self.avoid_steer_k) * (1.0 + 2.15 * emg)
        steer = np.tanh(steer_k * combined)
        base = float(self.avoid_base_speed)
        g = float(self.avoid_steer_gain)
        left_drive = base * (1.0 + g * steer)
        right_drive = base * (1.0 - g * steer)
        return np.array([left_drive, right_drive], dtype=float)

    def _apply_frontal_pivot(
        self,
        drives: np.ndarray,
        odor_drives: np.ndarray,
        avoid_drives: np.ndarray,
        feat: dict,
    ) -> np.ndarray:
        """Near-field spike wall: slow one tripod hard, speed the other (sharp re-aim)."""
        fp = float(feat.get("frontal_pressure", 0.0))
        emg = float(feat.get("emergency", 0.0))
        if fp <= 0.03 and emg <= 0.06:
            return drives

        pivot_signal = float(np.clip(fp + emg * 1.05, 0.0, 1.2))
        pm = float(np.clip(pivot_signal * float(self.pivot_mix_gain), 0.0, 0.92))

        h_l = float(feat["left_px"])
        h_r = float(feat["right_px"])
        hb_l = float(feat["left_bias"])
        hb_r = float(feat["right_bias"])

        escape = 0.0
        if abs(h_l - h_r) >= 11.0:
            escape = float(np.sign(h_l - h_r))
        elif abs(hb_l - hb_r) >= 0.085:
            escape = float(np.sign(hb_l - hb_r))

        if escape == 0.0:
            od = float(odor_drives[1] - odor_drives[0])
            if abs(od) > 0.055:
                escape = float(np.sign(od))
            else:
                av = float(avoid_drives[1] - avoid_drives[0])
                escape = float(np.sign(av)) if abs(av) > 0.055 else 1.0

        slow = float(self.pivot_drive_slow)
        fast = float(self.pivot_drive_fast)
        if escape > 0.0:
            pivot = np.array([fast, slow], dtype=float)
        else:
            pivot = np.array([slow, fast], dtype=float)

        out = (1.0 - pm) * np.asarray(drives, dtype=float) + pm * pivot
        return np.clip(out, 0.18, None)

    def _apply_collision_brake(self, drives: np.ndarray, feat: dict) -> np.ndarray:
        """Bleed speed when a spike fills the view; reduces momentum into geometry."""
        emg = float(feat.get("emergency", 0.0))
        if emg <= 0.12:
            return drives
        k = float(self.emergency_brake_strength) * np.power(emg, 1.1)
        scale = float(np.clip(1.0 - k, 0.55, 1.0))
        return np.clip(np.asarray(drives, dtype=float) * scale, 0.18, None)

    def _max_white_stack_height_and_cols_from_bottom(
        self, spikes_bin_u8: np.ndarray
    ) -> tuple[int, np.ndarray]:
        stack_h = self._max_vertical_white_run_lengths(spikes_bin_u8)
        if stack_h.size == 0:
            return 0, np.array([], dtype=int)
        m = int(stack_h.max(initial=0))
        cols = np.flatnonzero(stack_h == m)
        return m, cols

    def _max_vertical_white_run_lengths(self, spikes_bin_u8: np.ndarray) -> np.ndarray:
        """Per-column length of longest contiguous True segment (uint8 0/255 mask)."""
        m = np.asarray(spikes_bin_u8)
        if m.ndim != 2:
            raise ValueError(
                f"Expected spikes_bin as (H, W) uint8 mask; got shape {m.shape}"
            )
        h, w = m.shape
        if h <= 0 or w <= 0:
            return np.zeros((0,), dtype=int)
        bin01 = m > 0
        out = np.zeros(w, dtype=int)
        for c in range(w):
            col = bin01[:, c]
            cur = 0
            best = 0
            for v in col:
                if v:
                    cur += 1
                    if cur > best:
                        best = cur
                else:
                    cur = 0
            out[c] = best
        return out

    @staticmethod
    def _span_bottom_pref_for_run_length(
        col: np.ndarray, target_len: int
    ) -> tuple[int, int] | None:
        """Among True runs with length ``target_len``, pick the lowest span (closest to bottom)."""
        h = int(col.shape[0])
        best: tuple[int, int] | None = None
        i = 0
        while i < h:
            if not col[i]:
                i += 1
                continue
            j = i
            while j < h and col[j]:
                j += 1
            ln = j - i
            if ln == target_len:
                span = (i, j - 1)
                if best is None or span[1] > best[1]:
                    best = span
            i = j
        return best

    def _stack_columns_overlay_rgb(
        self,
        spikes_bin_u8: np.ndarray,
        max_stack_h: int,
        cols: np.ndarray,
        *,
        max_cols_draw: int = 12,
    ) -> np.ndarray:
        """RGB image: grayscale spikes_bin + red bars on winning columns."""
        g = np.asarray(spikes_bin_u8)
        if g.ndim != 2:
            raise ValueError(f"Expected (H, W) mask; got {g.shape}")
        h, w = g.shape
        rgb = np.stack([g, g, g], axis=-1).astype(np.uint8).copy()
        if max_stack_h <= 0 or cols.size == 0:
            return rgb

        cols_to_draw = np.asarray(cols).astype(int).reshape(-1)[:max_cols_draw]
        rcol = np.array([255, 0, 0], dtype=np.uint8)
        bin01 = g > 0

        y_mins: list[int] = []
        for c in cols_to_draw:
            if c < 0 or c >= w:
                continue
            span = self._span_bottom_pref_for_run_length(bin01[:, c], int(max_stack_h))
            if span is None:
                continue
            y0, y1 = span
            rgb[y0 : y1 + 1, c] = rcol
            y_mins.append(int(y0))

        if y_mins:
            y_top = min(y_mins)
            if 0 <= y_top < h:
                rgb[y_top, :] = rcol

        return rgb
