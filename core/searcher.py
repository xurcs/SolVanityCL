import logging
import time
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

from core.config import HostSetting
from core.opencl.manager import (
    get_all_gpu_devices,
    get_selected_gpu_devices,
)


class Searcher:
    def __init__(
        self,
        kernel_source: str,
        index: int,
        setting: HostSetting,
        chosen_devices: Optional[Tuple[int, List[int]]] = None,
    ):
        if chosen_devices is None:
            devices = get_all_gpu_devices()
        else:
            devices = get_selected_gpu_devices(*chosen_devices)
        enabled_device = devices[index]
        self.context = cl.Context([enabled_device])
        self.gpu_chunks = len(devices)
        self.command_queue = cl.CommandQueue(self.context)
        self.setting = setting
        self.index = index
        self.display_index = (
            index if chosen_devices is None else chosen_devices[1][index]
        )
        self.prev_time = None
        self.is_nvidia = "NVIDIA" in enabled_device.platform.name.upper()

        program = cl.Program(self.context, kernel_source).build()
        self.kernel = cl.Kernel(program, "generate_pubkey")
        self.memobj_key32 = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            32 * np.ubyte().itemsize,
            hostbuf=self.setting.key32,
        )
        self.memobj_output = cl.Buffer(
            self.context, cl.mem_flags.READ_WRITE, 33 * np.ubyte().itemsize
        )
        self.memobj_occupied_bytes = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.array([self.setting.iteration_bytes]),
        )
        self.memobj_group_offset = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.array([self.index]),
        )
        self.output = np.zeros(33, dtype=np.ubyte)
        self.kernel.set_arg(0, self.memobj_key32)
        self.kernel.set_arg(1, self.memobj_output)
        self.kernel.set_arg(2, self.memobj_occupied_bytes)
        self.kernel.set_arg(3, self.memobj_group_offset)

    def find(self, log_stats: bool = True) -> np.ndarray:
        start_time = time.time()
        copy_event = cl.enqueue_copy(self.command_queue, self.memobj_key32, self.setting.key32)
        
        global_worker_size = self.setting.global_work_size // self.gpu_chunks
        
        kernel_event = cl.enqueue_nd_range_kernel(
            self.command_queue,
            self.kernel,
            (global_worker_size,),
            (self.setting.local_work_size,),
            wait_for=[copy_event]
        )
        
        self.setting.increase_key32()

        read_event = cl.enqueue_copy(
            self.command_queue, 
            self.output, 
            self.memobj_output, 
            wait_for=[kernel_event]
        )
        read_event.wait()
        
        elapsed = time.time() - start_time
        self.prev_time = elapsed
        
        if log_stats:
            speed = global_worker_size / (elapsed * 1e6)
            logging.info(f"GPU {self.display_index} Speed: {speed:.2f} MH/s")
        
        return self.output


def multi_gpu_init(
    index: int,
    setting: HostSetting,
    gpu_counts: int,
    stop_flag,
    lock,
    chosen_devices: Optional[Tuple[int, List[int]]] = None,
) -> List:
    try:
        searcher = Searcher(
            kernel_source=setting.kernel_source,
            index=index,
            setting=setting,
            chosen_devices=chosen_devices,
        )
        
        i = 0
        st = time.time()
        batch_size = 5
        
        while True:
            # Process a batch of keys
            for _ in range(batch_size):
                result = searcher.find(i == 0)
                if result[0]:
                    with lock:
                        if not stop_flag.value:
                            stop_flag.value = 1
                    return result.tolist()
                i += 1
            
            current_time = time.time()
            if current_time - st > max(gpu_counts, 1):
                i = 0
                st = current_time
                with lock:
                    if stop_flag.value:
                        return result.tolist()
                        
    except Exception as e:
        logging.exception(e)
        
    return [0]


def save_result(outputs: List, output_dir: str) -> int:
    from core.utils.crypto import save_keypair
    from pathlib import Path
    import os

    keypairs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "keypairs")
    Path(keypairs_dir).mkdir(parents=True, exist_ok=True)
    result_count = 0
    for output in outputs:
        if not output[0]:
            continue
        result_count += 1
        pv_bytes = bytes(output[1:])
        save_keypair(pv_bytes, keypairs_dir)
    return result_count
