import secrets
from math import ceil

import numpy as np

DEFAULT_ITERATION_BITS = 24
DEFAULT_LOCAL_WORK_SIZE = 32


class HostSetting:
    def __init__(self, kernel_source: str, iteration_bits: int):
        self.iteration_bits = iteration_bits
        self.iteration_bytes = np.ubyte(ceil(iteration_bits / 8))
        self.global_work_size = 1 << iteration_bits
        self.local_work_size = 128
        self.kernel_source = kernel_source
        self.key32 = self.generate_key32()
        self._increment_value = 1 << iteration_bits

    def generate_key32(self) -> np.ndarray:
        key32 = np.zeros(32, dtype=np.ubyte)
        random_part = secrets.token_bytes(32 - int(self.iteration_bytes))
        key32[:32-int(self.iteration_bytes)] = np.frombuffer(random_part, dtype=np.ubyte)
        return key32

    def increase_key32(self) -> None:
        idx = 31
        carry = self._increment_value
        
        while carry > 0 and idx >= 0:
            total = int(self.key32[idx]) + carry
            self.key32[idx] = total & 0xFF
            carry = total >> 8
            idx -= 1
