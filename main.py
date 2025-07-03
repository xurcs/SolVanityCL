import multiprocessing
import os

from core.interactive import run_search

if __name__ == "__main__":
    os.environ["PYOPENCL_CTX"] = "0"
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"
    
    multiprocessing.set_start_method("spawn")
    
    try:
        os.nice(-10)
    except (AttributeError, OSError):
        pass
        
    run_search()
