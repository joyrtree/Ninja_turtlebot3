"""Small image-space lane tracker; no ROS, learned model or metric-ground claim.

All defaults are unmeasured trial settings, not validated robot limits.
Confidence describes observation quality, not a calibrated probability.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class FusionConfig:
    """Image fractions and seconds; max_speed/max_angular are trial SI commands."""
    max_speed: float = 0.10
    max_angular: float = 0.8
    kp: float = 1.6
    k_heading: float = 1.2
    kd: float = 0.0
    roi_top: float = 0.48
    roi_bottom: float = 0.94
    bands: int = 14
    min_track_points: int = 6
    band_half_height: float = 0.008
    min_luma: float = 70.0
    min_contrast: float = 11.0
    min_stripe_width: float = 0.002
    max_stripe_width: float = 0.085
    max_candidates_per_band: int = 8
    max_link_step: float = 0.09
    max_link_missed_bands: int = 2
    min_track_span: float = 0.23
    min_corridor_width: float = 0.18
    max_corridor_width: float = 0.99
    max_width_change: float = 0.40
    width_alpha: float = 0.25
    max_gap_s: float = 0.60
    max_track_age_s: float = 0.40
    max_width_age_s: float = 3.0
    association_distance: float = 0.12
    role_margin: float = 0.025
    ambiguity_margin: float = 0.035
    near_y: float = 0.86
    far_y: float = 0.64
    min_confidence: float = 0.32
    single_speed_factor: float = 0.55
    steering_slowdown: float = 1.8
    bend_slowdown: float = 4.0
    dt_min: float = 0.02
    dt_max: float = 0.25

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value):
                raise ValueError("Non-finite config: " + name)
        if not 0.0 < self.roi_top < self.far_y < self.near_y < self.roi_bottom <= 1.0:
            raise ValueError("Require roi_top < far_y < near_y < roi_bottom")
        if not 0 < self.min_corridor_width < self.max_corridor_width <= 1.0:
            raise ValueError("Invalid corridor width fractions")
        if not 0 < self.min_stripe_width < self.max_stripe_width < 0.5:
            raise ValueError("Invalid stripe width fractions")
        if self.bands < 6 or not 3 <= self.min_track_points <= self.bands:
            raise ValueError("Invalid band/track counts")
        if self.max_speed < 0 or self.max_angular <= 0:
            raise ValueError("Invalid trial command limits")
        if not 0 < self.width_alpha <= 1 or not 0 <= self.min_confidence < 1:
            raise ValueError("Invalid quality/width setting")
        if self.dt_min <= 0 or self.dt_max < self.dt_min:
            raise ValueError("Invalid derivative dt bounds")
        for name in ("max_gap_s", "max_track_age_s", "max_width_age_s", "association_distance"):
            if getattr(self, name) <= 0:
                raise ValueError("Require positive " + name)


@dataclass
class LaneResult:
    """Curves are optional Nx2 IMAGE pixels; quality is not correctness probability."""
    speed: float
    angular: float
    state: str
    confidence: float
    debug: dict
    left: Optional[np.ndarray] = None
    right: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None


class LaneFusion:
    """Track boundary roles, using validated image width only while it is recent.

    State is both/left_only/right_only, or a zero-command lost/ambiguous/invalid.
    Uninitialized single curves cannot identify their left/right role.
    """
    def __init__(self, config=None):
        self.config = config if config is not None else FusionConfig()
        self.reset()

    def reset(self):
        self._last_stamp = None
        self._track_stamp = None
        self._width_stamp = None
        self._left = None
        self._right = None
        self._width = None
        self._last_error = None
        self._shape = None

    def _stop(self, state, debug, confidence=0.0):
        self._last_error = None
        return LaneResult(0.0, 0.0, state, float(confidence), debug)

    def _candidates(self, image, ys):
        cfg = self.config
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        b, g, r = cv2.split(image.astype(np.float32))
        # A weak yellow hint augments contrast; color never sets boundary role.
        yellow = np.maximum(np.minimum(r, g) - b - 12.0, 0.0) / 100.0
        tracks = []
        half = max(1, int(round(cfg.band_half_height * height)))
        kernel = max(15, int(round(width * 0.075)) | 1)
        for row_index in range(len(ys) - 1, -1, -1):
            y = int(round(ys[row_index] * (height - 1)))
            lo, hi = max(0, y - half), min(height, y + half + 1)
            profile = np.mean(gray[lo:hi], axis=0)
            smooth = cv2.GaussianBlur(profile[None, :], (5, 1), 0)[0]
            background = cv2.GaussianBlur(smooth[None, :], (kernel, 1), 0)[0]
            contrast = smooth - background
            hint = np.clip(np.mean(yellow[lo:hi], axis=0), 0.0, 1.0)
            active = ((smooth >= cfg.min_luma) &
                      (contrast >= cfg.min_contrast * (1.0 - 0.18 * hint)))
            # Group both contrast edges belonging to ONE bright painted stripe.
            # Counting the two sides separately invents duplicate lane roles.
            edges = np.diff(np.r_[False, active, False].astype(np.int8))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            regions = []
            for start, end in zip(starts, ends):
                peak = int(start + np.argmax(smooth[start:end]))
                level = max(cfg.min_luma, float(smooth[peak]) - max(8.0, 1.5 * cfg.min_contrast))
                begin, finish = peak, peak + 1
                while begin > 0 and smooth[begin - 1] >= level:
                    begin -= 1
                while finish < width and smooth[finish] >= level:
                    finish += 1
                if cfg.min_stripe_width <= (finish - begin) / float(width) <= cfg.max_stripe_width:
                    regions.append((begin, finish))
            merged = []
            for begin, finish in sorted(regions):
                if merged and begin <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], finish))
                else:
                    merged.append((begin, finish))
            candidates = []
            for start, end in merged:
                stripe_width = (end - start) / float(width)
                if not cfg.min_stripe_width <= stripe_width <= cfg.max_stripe_width:
                    continue
                if not np.any(active[start:end]):
                    continue
                weights = np.maximum(contrast[start:end], 0.0) + 1.0
                x = float(np.average(np.arange(start, end), weights=weights)) / (width - 1)
                quality = min(1.0, float(np.percentile(contrast[start:end], 80)) / 60.0)
                quality *= 0.9 + 0.1 * float(np.mean(hint[start:end]))
                candidates.append((x, quality))
            candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:cfg.max_candidates_per_band]
            links = []
            for ti, track in enumerate(tracks):
                gap = track[-1][0] - row_index
                if gap < 1 or gap > cfg.max_link_missed_bands + 1:
                    continue
                prediction = track[-1][1]
                if len(track) >= 2:
                    old, recent = track[-2], track[-1]
                    slope = (recent[1] - old[1]) / (recent[0] - old[0])
                    prediction += np.clip(slope * (row_index - recent[0]), -0.10, 0.10)
                for ci, candidate in enumerate(candidates):
                    distance = abs(candidate[0] - prediction)
                    if distance <= cfg.max_link_step * gap:
                        links.append((distance, ti, ci))
            used_tracks, used_candidates = set(), set()
            for distance, ti, ci in sorted(links):
                if ti in used_tracks or ci in used_candidates:
                    continue
                tracks[ti].append((row_index, candidates[ci][0], candidates[ci][1]))
                used_tracks.add(ti)
                used_candidates.add(ci)
            for ci, candidate in enumerate(candidates):
                if ci not in used_candidates:
                    tracks.append([(row_index, candidate[0], candidate[1])])
        curves = []
        for track in tracks:
            if len(track) < cfg.min_track_points:
                continue
            points = sorted(track)
            indices = np.array([p[0] for p in points], dtype=int)
            observed_ys = ys[indices]
            if observed_ys[-1] - observed_ys[0] < cfg.min_track_span:
                continue
            if observed_ys[0] > cfg.far_y or observed_ys[-1] < cfg.near_y:
                continue
            xs = np.array([p[1] for p in points], dtype=float)
            # Piecewise interpolation preserves bends; no long temporal smoothing.
            curve = np.interp(ys, observed_ys, xs)
            coverage = len(points) / float(len(ys))
            quality = float(np.mean([p[2] for p in points]))
            quality = 0.55 * coverage + 0.45 * quality
            if np.isfinite(curve).all():
                curves.append({"x": curve, "quality": quality, "coverage": coverage})
        return curves

    def _pair_ok(self, left, right, ys):
        cfg = self.config
        widths = right - left
        if not np.isfinite(widths).all():
            return False
        if np.min(widths) < cfg.min_corridor_width or np.max(widths) > cfg.max_corridor_width:
            return False
        far_width = float(np.interp(cfg.far_y, ys, widths))
        near_width = float(np.interp(cfg.near_y, ys, widths))
        return 0.75 * far_width <= near_width <= 3.5 * far_width

    def process(self, image_bgr, stamp_s):
        cfg = self.config
        debug = {"quality_is_probability": False, "metric_ground_calibrated": False}
        try:
            stamp = float(stamp_s)
        except (TypeError, ValueError):
            self.reset()
            return self._stop("invalid", dict(debug, reason="invalid_stamp"))
        if not np.isfinite(stamp):
            self.reset()
            return self._stop("invalid", dict(debug, reason="nonfinite_stamp"))
        if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or
                image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8 or
                min(image_bgr.shape[:2]) < 60):
            self.reset()
            return self._stop("invalid", dict(debug, reason="expect_uint8_bgr_image"))
        shape = image_bgr.shape[:2]
        if self._shape is not None and shape != self._shape:
            self.reset()
            debug["geometry_reset"] = True
        self._shape = shape
        raw_dt = stamp - self._last_stamp if self._last_stamp is not None else 0.0
        if self._last_stamp is not None and raw_dt <= 0:
            self.reset()
            return self._stop("invalid", dict(debug, reason="nonincreasing_stamp", dt_raw=raw_dt))
        if self._last_stamp is not None and raw_dt > cfg.max_gap_s:
            self.reset()
            self._shape = shape
            debug["gap_reset"] = True
        self._last_stamp = stamp
        if self._track_stamp is not None and stamp - self._track_stamp > cfg.max_gap_s:
            self._left = self._right = self._width = None
            self._track_stamp = self._width_stamp = None
            self._last_error = None
            debug["track_expired"] = True
        dt = float(np.clip(raw_dt if raw_dt > 0 else cfg.dt_min, cfg.dt_min, cfg.dt_max))
        debug.update(stamp_s=stamp, dt_raw=float(raw_dt), dt_bounded=dt)
        ys = np.linspace(cfg.roi_top, cfg.roi_bottom, cfg.bands)
        curves = self._candidates(image_bgr, ys)
        debug["curve_candidates"] = len(curves)
        if not curves:
            return self._stop("lost", dict(debug, reason="no_valid_curve"))
        left = right = center = None
        confidence = 0.0
        state = "ambiguous"
        recent = (self._track_stamp is not None and
                  stamp - self._track_stamp <= cfg.max_track_age_s and
                  self._width is not None)
        if recent:
            matches = []
            for ci, curve in enumerate(curves):
                dl = float(np.mean(np.abs(curve["x"] - self._left)))
                dr = float(np.mean(np.abs(curve["x"] - self._right)))
                if min(dl, dr) <= cfg.association_distance and abs(dl - dr) >= cfg.role_margin:
                    matches.append((min(dl, dr), ci, "left" if dl < dr else "right"))
            chosen = {}
            for distance, ci, role in sorted(matches):
                if role not in chosen:
                    chosen[role] = (curves[ci], distance)
            if "left" in chosen and "right" in chosen:
                lc, rc = chosen["left"][0], chosen["right"][0]
                if self._pair_ok(lc["x"], rc["x"], ys):
                    widths = rc["x"] - lc["x"]
                    relative_change = float(np.max(np.abs(widths - self._width) / np.maximum(self._width, 0.01)))
                    if relative_change <= cfg.max_width_change:
                        left, right = lc["x"], rc["x"]
                        self._width = (1.0 - cfg.width_alpha) * self._width + cfg.width_alpha * widths
                        self._width_stamp = stamp
                        confidence = min(lc["quality"], rc["quality"])
                        state = "both"
            elif len(chosen) == 1 and self._width_stamp is not None and stamp - self._width_stamp <= cfg.max_width_age_s:
                role = next(iter(chosen))
                curve, distance = chosen[role]
                if role == "left":
                    left = curve["x"]
                    right = left + self._width
                    state = "left_only"
                else:
                    right = curve["x"]
                    left = right - self._width
                    state = "right_only"
                confidence = curve["quality"] * 0.80 * max(0.5, 1.0 - distance / cfg.association_distance)
        else:
            pairs = []
            for li, lc in enumerate(curves):
                for ri, rc in enumerate(curves):
                    if li == ri or not self._pair_ok(lc["x"], rc["x"], ys):
                        continue
                    ln = float(np.interp(cfg.near_y, ys, lc["x"]))
                    rn = float(np.interp(cfg.near_y, ys, rc["x"]))
                    if not ln < 0.5 < rn:
                        continue
                    middle = (lc["x"] + rc["x"]) * 0.5
                    score = abs((ln + rn) * 0.5 - 0.5) + 0.08 * (2.0 - lc["quality"] - rc["quality"])
                    pairs.append((score, li, ri, middle))
            pairs.sort(key=lambda item: item[0])
            if pairs:
                best = pairs[0]
                ambiguous = False
                for alternative in pairs[1:]:
                    if alternative[0] - best[0] > cfg.ambiguity_margin:
                        break
                    if (np.mean(np.abs(alternative[3] - best[3])) > 0.04 or
                            alternative[1:3] != best[1:3]):
                        ambiguous = True
                        break
                if not ambiguous:
                    lc, rc = curves[best[1]], curves[best[2]]
                    left, right = lc["x"], rc["x"]
                    self._width = right - left
                    self._width_stamp = stamp
                    confidence = min(lc["quality"], rc["quality"])
                    state = "both"
                else:
                    debug["reason"] = "multiple_plausible_corridors"
        if left is None or right is None:
            return self._stop("ambiguous", dict(debug, reason=debug.get("reason", "no_independent_boundary_role_or_recent_width")))
        center = (left + right) * 0.5
        if (not np.isfinite(center).all() or np.any(center < 0.02) or np.any(center > 0.98) or
                confidence < cfg.min_confidence):
            return self._stop("ambiguous", dict(debug, reason="invalid_or_low_quality_path"), confidence)
        self._left, self._right = left.copy(), right.copy()
        self._track_stamp = stamp
        near_x = float(np.interp(cfg.near_y, ys, center))
        far_x = float(np.interp(cfg.far_y, ys, center))
        lateral_error = (near_x - 0.5) * 2.0
        heading_error = (far_x - near_x) * 2.0
        derivative = 0.0 if self._last_error is None else (lateral_error - self._last_error) / dt
        self._last_error = lateral_error
        angular = float(np.clip(-(cfg.kp * lateral_error + cfg.k_heading * heading_error + cfg.kd * derivative),
                                -cfg.max_angular, cfg.max_angular))
        bend_proxy = float(np.max(np.abs(np.diff(center, n=2)))) if len(center) >= 3 else 0.0
        quality_scale = float(np.clip(confidence, 0.0, 1.0))
        speed = cfg.max_speed * quality_scale / (1.0 + cfg.steering_slowdown * abs(angular) / cfg.max_angular + cfg.bend_slowdown * bend_proxy)
        if state != "both":
            speed *= cfg.single_speed_factor
        debug.update(lateral_error=lateral_error, heading_error=heading_error,
                     derivative=derivative, image_bend_proxy=bend_proxy,
                     width_age_s=float(stamp - self._width_stamp),
                     width_near_fraction=float(np.interp(cfg.near_y, ys, self._width)),
                     left_observed=state in ("both", "left_only"),
                     right_observed=state in ("both", "right_only"))
        height, width = shape
        points = lambda xs: np.column_stack((xs * (width - 1), ys * (height - 1))).astype(np.float32)
        return LaneResult(float(speed), angular, state, float(confidence), debug,
                          points(left), points(right), points(center))


def draw_result(image, result):
    """Return a copy; inferred boundaries are labeled by the state/debug flags."""
    canvas = image.copy()
    for name, color in (("left", (40, 210, 255)), ("right", (255, 170, 70)), ("center", (70, 230, 70))):
        curve = getattr(result, name)
        if curve is not None and np.isfinite(curve).all():
            cv2.polylines(canvas, [np.round(curve).astype(np.int32)], False, color, 2, cv2.LINE_AA)
    height, width = canvas.shape[:2]
    cv2.line(canvas, (width // 2, int(height * 0.72)), (width // 2, height - 1), (180, 180, 180), 1)
    lines = ["%s  quality=%.2f (proxy)" % (result.state, result.confidence),
             "trial v=%.3f m/s w=%.3f rad/s" % (result.speed, result.angular)]
    if result.state in ("left_only", "right_only"):
        lines.append("ONE boundary inferred from recent width")
    if result.state in ("lost", "ambiguous", "invalid"):
        lines.append("NO VALID PATH: zero command")
    for index, content in enumerate(lines):
        cv2.putText(canvas, content, (8, 23 + index * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(canvas, content, (8, 23 + index * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas
