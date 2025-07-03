import logging
import multiprocessing
import sys
from multiprocessing.pool import Pool
from typing import List, Optional, Tuple

import pyopencl as cl

from core.config import DEFAULT_ITERATION_BITS, HostSetting
from core.opencl.manager import (
    get_all_gpu_devices,
    get_chosen_devices,
)
from core.searcher import multi_gpu_init, save_result
from core.utils.helpers import check_character, load_kernel_source

logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")

def prompt(message, default=None, validate=None):
    while True:
        if default is not None:
            user_input = input(f"{message} [{default}]: ").strip()
            if not user_input:
                user_input = default
        else:
            user_input = input(f"{message}: ").strip()
        
        if validate and user_input:
            try:
                validate(user_input)
                return user_input
            except Exception as e:
                print(f"Error: {e}")
        else:
            return user_input

def run_search():
    keypairs_dir = "keypairs"
    
    starts_with = prompt("Enter prefix for public key")
    if starts_with:
        check_character("starts_with", starts_with)
    
    ends_with = prompt("Enter suffix for public key")
    if ends_with:
        check_character("ends_with", ends_with)
    
    if not starts_with and not ends_with:
        print("Please provide at least one prefix or suffix.")
        return
    
    try:
        count = int(prompt("Number of keys to generate", "1"))
        if count <= 0:
            count = 1
    except ValueError:
        count = 1
    
    try:
        iteration_bits = int(prompt("Iteration bits", str(DEFAULT_ITERATION_BITS)))
    except ValueError:
        iteration_bits = DEFAULT_ITERATION_BITS
    
    is_case_sensitive = prompt("Case sensitive? (y/n)", "y").lower() == "y"
    
    select_device = prompt("Select OpenCL device manually? (y/n)", "n").lower() == "y"
    
    chosen_devices: Optional[Tuple[int, List[int]]] = None
    if select_device:
        chosen_devices = get_chosen_devices()
        gpu_counts = len(chosen_devices[1])
    else:
        gpu_counts = len(get_all_gpu_devices())

    logging.info(
        "Searching Solana pubkey with starts_with=%s, ends_with=%s, is_case_sensitive=%s",
        repr(starts_with),
        repr(ends_with),
        is_case_sensitive,
    )
    logging.info(f"Using {gpu_counts} OpenCL device(s)")

    result_count = 0
    with multiprocessing.Manager() as manager:
        with Pool(processes=gpu_counts) as pool:
            kernel_source = load_kernel_source(
                (starts_with,) if starts_with else (), ends_with, is_case_sensitive
            )
            lock = manager.Lock()
            while result_count < count:
                stop_flag = manager.Value("i", 0)
                results = pool.starmap(
                    multi_gpu_init,
                    [
                        (
                            x,
                            HostSetting(kernel_source, iteration_bits),
                            gpu_counts,
                            stop_flag,
                            lock,
                            chosen_devices,
                        )
                        for x in range(gpu_counts)
                    ],
                )
                result_count += save_result(results, keypairs_dir)
