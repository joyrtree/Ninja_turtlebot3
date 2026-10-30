"""Meaningful scheduling/deadline checks without ROS or a robot."""
import threading
import time
from types import SimpleNamespace

from lane_detection_fusion.runtime import LatestFrameRuntime, RuntimeConfig


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


def result(speed=0.1, angular=0.2, state="TRACK"):
    return SimpleNamespace(speed=speed, angular=angular, state=state, confidence=0.8)


def make_runtime(processor=None, config=None):
    clock = Clock()
    commands = []
    seen = []

    def process(job):
        seen.append(job)
        return processor(job, clock) if processor else result()

    runtime = LatestFrameRuntime(process, commands.append,
                                 config or RuntimeConfig(), clock, clock, autostart=False)
    runtime.set_enabled(True)
    return runtime, clock, commands, seen


def test_only_latest_pending_frame_is_processed():
    runtime, clock, commands, seen = make_runtime()
    runtime.submit("old", clock())
    runtime.submit("new", clock())
    runtime.process_next()
    assert [job.payload for job in seen] == ["new"]
    assert runtime.stats["replaced_pending"] == 1
    assert commands[-1].reason == "fresh_result"


def test_stale_input_is_rejected_before_processing():
    runtime, clock, commands, seen = make_runtime()
    runtime.submit("image", clock())
    clock.value += 0.26
    runtime.process_next()
    assert seen == []
    assert commands[-1].speed == commands[-1].angular == 0
    assert commands[-1].reason == "input_stale"


def test_slow_inference_result_is_rejected_after_processing():
    def slow(job, clock):
        clock.value += 0.31
        return result()
    runtime, clock, commands, seen = make_runtime(slow)
    runtime.submit("image", clock())
    runtime.process_next()
    assert len(seen) == 1
    assert commands[-1].reason == "result_stale"
    assert not any(c.speed > 0 for c in commands)


def test_header_age_is_checked_before_and_after_inference():
    runtime, clock, commands, seen = make_runtime()
    runtime.submit("old_stamp", clock() - 1.0)
    runtime.process_next()
    assert not seen
    assert commands[-1].reason == "header_stale"
    mono = Clock()
    ros = Clock()
    published = []
    def process(job):
        ros.value += 0.4
        return result()
    runtime = LatestFrameRuntime(process, published.append, RuntimeConfig(), mono, ros, False)
    runtime.set_enabled(True)
    runtime.submit("new_stamp", ros())
    runtime.process_next()
    assert published[-1].reason == "header_stale"


def test_disable_during_inference_cannot_be_overwritten():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    commands = []
    def process(job):
        entered.set()
        release.wait(2)
        finished.set()
        return result()
    runtime = LatestFrameRuntime(process, commands.append,
                                 RuntimeConfig(check_header_age=False))
    try:
        runtime.set_enabled(True)
        runtime.submit("image")
        assert entered.wait(1)
        runtime.set_enabled(False)
        disabled_index = len(commands) - 1
        assert commands[-1].reason == "disabled"
        release.set()
        assert finished.wait(1)
        # Wait until the worker records completion, not a hardware timing test.
        deadline = time.monotonic() + 1
        while runtime.stats["disabled"] == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert all(c.speed == c.angular == 0 for c in commands[disabled_index:])
    finally:
        release.set()
        runtime.close()


def test_mode_generation_rejects_a_running_job():
    runtime = None
    def process(job, clock):
        runtime.set_mode_hint(True)
        return result()
    runtime, clock, commands, seen = make_runtime(process)
    runtime.submit("image", clock())
    runtime.process_next()
    assert commands[-1].reason == "mode_changed"
    assert runtime.stats["generation_changed"] == 1
    assert not any(c.speed > 0 for c in commands)


def test_reset_generation_rejects_a_running_job():
    runtime = None
    def process(job, clock):
        runtime.reset()
        return result()
    runtime, clock, commands, seen = make_runtime(process)
    runtime.submit("image", clock())
    runtime.process_next()
    assert commands[-1].reason == "reset"
    assert not any(c.speed > 0 for c in commands)


def test_watchdog_uses_original_receipt_not_republication_time():
    runtime, clock, commands, seen = make_runtime(config=RuntimeConfig(check_header_age=False))
    runtime.submit("image", clock())
    runtime.process_next()
    clock.value += 0.31
    runtime.watchdog()
    assert commands[-1].reason == "command_expired"
    assert commands[-1].speed == commands[-1].angular == 0
    count = len(commands)
    runtime.watchdog()
    assert len(commands) == count


def test_watchdog_stops_while_worker_is_blocked():
    entered, release = threading.Event(), threading.Event()
    commands = []
    clock = Clock()
    def process(job):
        entered.set()
        release.wait(2)
        return result()
    runtime = LatestFrameRuntime(process, commands.append,
                                 RuntimeConfig(check_header_age=False), clock)
    try:
        runtime.set_enabled(True)
        runtime.submit("image")
        assert entered.wait(1)
        clock.value += 0.4
        runtime.watchdog()
        assert commands[-1].reason == "input_silence"
        assert commands[-1].speed == commands[-1].angular == 0
    finally:
        release.set()
        runtime.close()


def test_backward_ros_clock_jump_invalidates_result():
    runtime, clock, commands, seen = make_runtime()
    runtime.submit("first", clock())
    runtime.process_next()
    clock.value -= 1
    runtime.submit("after_jump", clock())
    runtime.process_next()
    assert commands[-1].reason == "ros_clock_backward"
    assert commands[-1].speed == 0


def test_lost_result_cannot_rotate_or_move():
    runtime, clock, commands, seen = make_runtime(lambda job, clock: result(0.2, 0.5, "LOST"))
    runtime.submit("image", clock())
    runtime.process_next()
    assert commands[-1].reason == "lane_lost"
    assert commands[-1].speed == commands[-1].angular == 0


def test_watchdog_detects_ros_clock_jump_without_another_image():
    runtime, clock, commands, seen = make_runtime()
    runtime.submit("image", clock())
    runtime.process_next()
    clock.value -= 1.0
    runtime.watchdog()
    assert commands[-1].reason == "ros_clock_backward"
    assert commands[-1].speed == commands[-1].angular == 0


def test_command_with_old_header_expires_before_receipt_deadline():
    mono, ros = Clock(), Clock()
    commands = []
    runtime = LatestFrameRuntime(lambda job: result(), commands.append,
                                 RuntimeConfig(), mono, ros, False)
    runtime.set_enabled(True)
    runtime.submit("arrived_late", ros() - 0.24)
    runtime.process_next()
    assert commands[-1].reason == "fresh_result"
    assert commands[-1].stamp_s == 99.76
    # Receipt age is only .07s, but the observation is already .31s old.
    mono.value += 0.07
    ros.value += 0.07
    runtime.watchdog()
    assert commands[-1].reason == "command_header_stale"
    assert commands[-1].speed == commands[-1].angular == 0


def test_forward_ros_clock_jump_expires_command_without_another_image():
    mono, ros = Clock(), Clock()
    commands = []
    runtime = LatestFrameRuntime(lambda job: result(), commands.append,
                                 RuntimeConfig(), mono, ros, False)
    runtime.set_enabled(True)
    runtime.submit("image", ros())
    runtime.process_next()
    ros.value += 1.0
    runtime.watchdog()
    assert commands[-1].reason == "command_header_stale"
    assert commands[-1].speed == commands[-1].angular == 0


def test_configuration_rejects_bad_deadlines():
    for kwargs in ({"max_result_age_s": 0}, {"control_hz": True},
                   {"max_frame_age_s": float("nan")},
                   {"clock_tolerance_s": -1}, {"intersection_speed_factor": 2}):
        try:
            RuntimeConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid configuration accepted")
