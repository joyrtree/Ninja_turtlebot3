"""ROS-independent latest-frame execution and command validity checks."""
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RuntimeConfig:
    max_frame_age_s: float = 0.25
    max_result_age_s: float = 0.30
    max_input_silence_s: float = 0.35
    control_hz: float = 20.0
    check_header_age: bool = True
    clock_tolerance_s: float = 0.05
    intersection_speed_factor: float = 0.7

    def __post_init__(self):
        for name in ("max_frame_age_s", "max_result_age_s",
                     "max_input_silence_s", "control_hz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        if not isinstance(self.check_header_age, bool):
            raise ValueError("check_header_age must be a bool")
        if not math.isfinite(self.clock_tolerance_s) or self.clock_tolerance_s < 0:
            raise ValueError("clock_tolerance_s must be finite and nonnegative")
        if not 0 < self.intersection_speed_factor <= 1:
            raise ValueError("intersection_speed_factor must be in (0, 1]")


@dataclass(frozen=True)
class FrameJob:
    payload: Any
    stamp_s: Optional[float]
    received_monotonic: float
    generation: int
    sequence: int
    mode_hint: bool
    is_intersection: bool


@dataclass(frozen=True)
class Command:
    speed: float
    angular: float
    state: str
    reason: str
    generation: int
    sequence: int = -1
    received_monotonic: Optional[float] = None
    confidence: float = 0.0
    result: Any = None
    stamp_s: Optional[float] = None


class LatestFrameRuntime:
    """One pending input, one inference worker, independent wall-time watchdog.

    ``processor(job)`` is called without the state lock. ``publish(command)`` is
    called under that lock and must be short. Thus a disable zero is ordered
    after any already-started publication and before any later completion.
    Core reset belongs in the worker when job.generation changes, not in ROS
    callbacks. Autostart=False exposes process_next/watchdog for deterministic
    tests; the same validity paths run in the threaded mode.
    """
    def __init__(self, processor, publish, config=None, monotonic=None,
                 ros_now=None, autostart=True):
        self.config = config or RuntimeConfig()
        self._processor = processor
        self._publish = publish
        self._mono = monotonic or time.monotonic
        self._ros_now = ros_now
        if self.config.check_header_age and ros_now is None:
            raise ValueError("ros_now required when check_header_age=True")
        self._condition = threading.Condition(threading.RLock())
        self._stop = threading.Event()
        self._closed = False
        self._enabled = False
        self._mode_hint = False
        self._intersection = False
        self._generation = 0
        self._sequence = 0
        self._pending = None
        self._last_input = None
        self._last_valid = None
        self._last_ros = None
        self._last_reason = None
        self.stats = Counter()
        self.last_error = ""
        self._threads = []
        if autostart:
            self._threads = [
                threading.Thread(target=self._worker_loop, name="lane-inference", daemon=True),
                threading.Thread(target=self._watchdog_loop, name="lane-watchdog", daemon=True),
            ]
            for thread in self._threads:
                thread.start()

    @property
    def generation(self):
        with self._condition:
            return self._generation

    def _send_locked(self, command):
        try:
            self._publish(command)
            self.stats["commands"] += 1
        except Exception as exc:
            self.last_error = "publish: %s" % exc
            self.stats["publish_errors"] += 1

    def _zero_locked(self, reason, force=False):
        self._last_valid = None
        if force or self._last_reason != reason:
            self._send_locked(Command(0.0, 0.0, "STOP", reason, self._generation))
        self._last_reason = reason

    def _invalidate_locked(self, reason):
        self._generation += 1
        self._pending = None
        self._last_input = None
        self._zero_locked(reason, force=True)
        self._condition.notify_all()

    def set_enabled(self, enabled):
        with self._condition:
            if self._closed:
                return
            enabled = bool(enabled)
            if enabled != self._enabled:
                self._enabled = enabled
                self._invalidate_locked("enabled_waiting_input" if enabled else "disabled")
            elif not enabled:
                self._zero_locked("disabled", force=True)

    def set_mode_hint(self, mode):
        with self._condition:
            if not self._closed and bool(mode) != self._mode_hint:
                self._mode_hint = bool(mode)
                self._invalidate_locked("mode_changed")

    def set_intersection(self, enabled):
        with self._condition:
            if not self._closed and bool(enabled) != self._intersection:
                self._intersection = bool(enabled)
                self._invalidate_locked("intersection_changed")

    def reset(self):
        with self._condition:
            if not self._closed:
                self._invalidate_locked("reset")

    def submit(self, payload, stamp_s=None):
        now = self._mono()
        with self._condition:
            if self._closed or not self._enabled:
                self.stats["inactive_inputs"] += 1
                return False
            if self._pending is not None:
                self.stats["replaced_pending"] += 1
            self._sequence += 1
            self._pending = FrameJob(payload, stamp_s, now, self._generation,
                                     self._sequence, self._mode_hint, self._intersection)
            self._last_input = now
            self.stats["inputs"] += 1
            self._condition.notify()
            return True

    def _ros_reason_locked(self, job):
        if not self.config.check_header_age:
            return None
        now = float(self._ros_now())
        if not math.isfinite(now):
            return "invalid_ros_clock"
        if self._last_ros is not None and now < self._last_ros - self.config.clock_tolerance_s:
            self._last_ros = now
            self._invalidate_locked("ros_clock_backward")
            return "ros_clock_backward"
        self._last_ros = now
        stamp = job.stamp_s
        if stamp is None or not math.isfinite(stamp) or stamp <= 0:
            return "invalid_header_stamp"
        age = now - stamp
        if age < -self.config.clock_tolerance_s:
            return "header_in_future"
        if age > self.config.max_result_age_s:
            return "header_stale"
        return None

    def _reason_locked(self, job, before=False):
        if self._closed:
            return "closed"
        if not self._enabled:
            return "disabled"
        if job.generation != self._generation:
            return "generation_changed"
        limit = self.config.max_frame_age_s if before else self.config.max_result_age_s
        if self._mono() - job.received_monotonic > limit:
            return "input_stale" if before else "result_stale"
        reason = self._ros_reason_locked(job)
        if reason == "header_stale" and before:
            return reason
        if before and reason is None and self.config.check_header_age:
            if float(self._ros_now()) - job.stamp_s > self.config.max_frame_age_s:
                return "header_stale"
        return reason

    def process_next(self):
        with self._condition:
            job = self._pending
            self._pending = None
            if job is None or self._closed:
                return False
            reason = self._reason_locked(job, before=True)
            if reason:
                self.stats[reason] += 1
                if not self._closed and reason != "generation_changed":
                    self._zero_locked(reason)
                return True
        try:
            result = self._processor(job)
            error = None
        except Exception as exc:
            result = None
            error = "%s: %s" % (type(exc).__name__, exc)
        with self._condition:
            reason = self._reason_locked(job)
            if reason:
                self.stats[reason] += 1
                # An old completion must not overwrite a newer generation's zero
                # or valid result, including during shutdown.
                if reason not in ("closed", "generation_changed", "disabled"):
                    self._zero_locked(reason)
                return True
            if error is not None:
                self.last_error = error
                self.stats["inference_errors"] += 1
                self._zero_locked("inference_error")
                return True
            try:
                speed, angular = float(result.speed), float(result.angular)
                confidence = float(result.confidence)
                state = str(result.state)
                if not all(math.isfinite(v) for v in (speed, angular, confidence)) or speed < 0:
                    raise ValueError("nonfinite/negative command")
            except (ValueError, TypeError, AttributeError) as exc:
                self.last_error = str(exc)
                self._zero_locked("invalid_result")
                return True
            if speed <= 0 or any(word in state.lower() for word in ("lost", "invalid", "ambiguous", "disabled")):
                self.stats["lost_results"] += 1
                self._zero_locked("lane_lost")
                return True
            if job.is_intersection:
                speed *= self.config.intersection_speed_factor
                angular *= self.config.intersection_speed_factor
            command = Command(speed, angular, state, "fresh_result", job.generation,
                              job.sequence, job.received_monotonic, confidence, result,
                              stamp_s=job.stamp_s)
            self._last_valid = command
            self._last_reason = "fresh_result"
            self._send_locked(command)
            self.stats["processed"] += 1
        return True

    def watchdog(self):
        with self._condition:
            if self._closed:
                return
            if not self._enabled:
                self._zero_locked("disabled")
                return
            if self.config.check_header_age:
                ros_now = float(self._ros_now())
                if not math.isfinite(ros_now):
                    self._invalidate_locked("invalid_ros_clock")
                    return
                if self._last_ros is not None and ros_now < self._last_ros - self.config.clock_tolerance_s:
                    self._last_ros = ros_now
                    self._invalidate_locked("ros_clock_backward")
                    return
                self._last_ros = ros_now
                if self._last_valid is not None:
                    stamp = self._last_valid.stamp_s
                    if stamp is None or not math.isfinite(stamp):
                        self._zero_locked("command_invalid_header")
                        return
                    header_age = ros_now - stamp
                    if header_age > self.config.max_result_age_s:
                        self._zero_locked("command_header_stale")
                        return
                    if header_age < -self.config.clock_tolerance_s:
                        self._invalidate_locked("command_header_future")
                        return
            now = self._mono()
            if self._last_input is None:
                self._zero_locked("waiting_input")
            elif now - self._last_input > self.config.max_input_silence_s:
                self._zero_locked("input_silence")
            elif self._last_valid is not None:
                if now - self._last_valid.received_monotonic > self.config.max_result_age_s:
                    self._zero_locked("command_expired")

    def _worker_loop(self):
        while not self._stop.is_set():
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closed)
                if self._closed:
                    return
            self.process_next()

    def _watchdog_loop(self):
        while not self._stop.wait(1.0 / self.config.control_hz):
            self.watchdog()

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._enabled = False
            self._generation += 1
            self._pending = None
            self._zero_locked("shutdown", force=True)
            self._closed = True
            self._stop.set()
            self._condition.notify_all()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=0.5)
