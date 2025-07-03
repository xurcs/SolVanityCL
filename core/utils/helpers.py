import logging
import platform
from pathlib import Path
from typing import Tuple

import pyopencl as cl
from base58 import b58decode


def check_character(name: str, character: str) -> None:
    try:
        b58decode(character)
    except ValueError as e:
        logging.error(f"{str(e)} in {name}")
        raise SystemExit(1)
    except Exception as e:
        raise e


_kernel_source_cache = {}

def load_kernel_source(
    starts_with_list: Tuple[str], ends_with: str, is_case_sensitive: bool
) -> str:
    cache_key = (tuple(starts_with_list), ends_with, is_case_sensitive)
    
    if cache_key in _kernel_source_cache:
        return _kernel_source_cache[cache_key]
        
    prefixes = []
    if starts_with_list:
        prefixes = [list(prefix.encode()) for prefix in starts_with_list]
    else:
        prefixes = [[]]
        
    max_prefix_len = max((len(p) for p in prefixes), default=0)
    
    for p in prefixes:
        p.extend([0] * (max_prefix_len - len(p)))
    
    SUFFIX_BYTES = list(ends_with.encode())
    
    kernel_path = Path(__file__).parent.parent / "opencl" / "kernel.cl"
    if not kernel_path.exists():
        raise FileNotFoundError("Kernel source file not found.")
    
    with kernel_path.open("r") as f:
        source = f.read()
    
    replacements = {
        "#define N ": f"#define N {len(prefixes)}\n",
        "#define L ": f"#define L {max_prefix_len}\n",
    }
    
    for pattern, replacement in replacements.items():
        idx = source.find(pattern)
        if idx >= 0:
            end_idx = source.find("\n", idx)
            source = source[:idx] + replacement + source[end_idx+1:]
    
    prefixes_str = "{"
    for prefix in prefixes:
        prefixes_str += "{" + ", ".join(map(str, prefix)) + "}, "
    prefixes_str = prefixes_str.rstrip(", ") + "}"
    source = source.replace(
        "constant uchar PREFIXES[N][L] = {{83, 111, 76}};", 
        f"constant uchar PREFIXES[N][L] = {prefixes_str};"
    )
    source = source.replace(
        "constant uchar SUFFIX[] = {};", 
        f"constant uchar SUFFIX[] = {{{', '.join(map(str, SUFFIX_BYTES))}}};"
    )
    source = source.replace(
        "constant bool CASE_SENSITIVE = true;",
        f"constant bool CASE_SENSITIVE = {str(is_case_sensitive).lower()};"
    )
    
    if "NVIDIA" in str(cl.get_platforms()) and platform.system() == "Windows":
        source = source.replace("#define __generic\n", "")
    if cl.get_cl_header_version()[0] != 1 and platform.system() != "Windows":
        source = source.replace("#define __generic\n", "")
    
    _kernel_source_cache[cache_key] = source
    return source
