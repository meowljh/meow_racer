from collections import defaultdict
from typing import Callable

import numpy as np

class Accumulator:
    __slots__ = ("prefix", "buffer")

    def __init__(self, prefix=""):
        self.prefix = prefix
        self.buffer = defaultdict(list)

    def add(self, key, value):
        self.buffer[key].append(value)

    def add_vec(self, key, value): #for vectorized env
        self.buffer[key].extend(value)

    def add_all(self, data: dict):
        for key, value in data.items():
            self.add(key, value)

    def reset(self):
        self.buffer.clear()

    def log(self, log_fn: Callable[[str, float], None]):
        for key, values in self.buffer.items():
            key = key if not self.prefix else f"{self.prefix}/{key}"
            value = sum(values) / len(values)
            log_fn(key, value)

class UpdateLog:
    __slots__ = ("update_step", "accumulator")

    def __init__(self):
        self.update_step = 0
        self.accumulator = Accumulator("update")

    def add(self, metrics: dict):
        self.update_step += 1
        self.accumulator.add_all(metrics)

    def log(self, log_fn: Callable[[str, float, int], None]):
        self.accumulator.log(lambda k, v: log_fn(k, v, self.update_step * 5))
        self.accumulator.reset()


class Interval:
    __slots__ = ("interval", "last_step")

    def __init__(self, interval: int):
        self.interval = interval
        self.last_step = 0

    def check(self, step: int) -> bool:
        if step - self.last_step >= self.interval:
            self.last_step = step
            return True
        return False


class SampleLog:
    __slots__ = ("sample_step", "sample_episode", "episode_len", "episode_ret", "accumulator")


    def __init__(self):
        super().__init__()
        self.sample_step = 0 #total number of exploration steps tracked (NOT reset to 0)
        self.sample_episode = 0
        self.episode_len = 0 #total length of steps for the current episode
        self.episode_ret = 0. #return summed reward value for the current episode
        self.accumulator = Accumulator("sample")
    
    def add(self, reward:float, terminated:bool, truncated:bool):
        self.episode_ret += reward
        self.episode_len += 1
        self.sample_step += 1

        done = truncated or terminated
        if done:
            self.sample_episode += 1
            self.accumulator.add("episode_ret", float(self.episode_ret))
            self.accumulator.add("episode_len", int(self.episode_len))
            self.episode_len = 0
            self.episode_ret = 0.
        
        return done

