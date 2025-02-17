
import os
import psutil
import tracemalloc

class MemUtil(object):
    rss_mem_init = None
    rss_mem_prev = None
    python_mem_init = None
    python_mem_prev = None

    @staticmethod
    def print_rss_memory_usage():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        current = mem_info.rss / (1024 ** 2) # to MegaByte
        if MemUtil.rss_mem_init is None:
            MemUtil.rss_mem_init = current
            MemUtil.rss_mem_prev = current
        print(f"RSS Mem Usage: cur={current:.2f} incr={current - MemUtil.rss_mem_prev:.2f} "
                f"acc_incr={current - MemUtil.rss_mem_init:.2f} MB")
        MemUtil.rss_mem_prev = current

    @staticmethod
    def start_python_memory_tracking():
        tracemalloc.start()

    @staticmethod
    def stop_python_memory_tracking():
        tracemalloc.stop()

    @staticmethod
    def print_python_memory_usage():
        current, peak = tracemalloc.get_traced_memory()
        current = current / (1024 ** 2) # to MegaByte
        peak = peak / (1024 ** 2) # to MegaByte
        if MemUtil.python_mem_init is None:
            MemUtil.python_mem_init = current
            MemUtil.python_mem_prev = current
        print(f"Python Mem Usage: cur={current:.2f} incr={current - MemUtil.python_mem_prev:.2f} "
                f"acc_incr={current - MemUtil.python_mem_init:.2f} peak={peak:.2f} MB")
        MemUtil.python_mem_prev = current

    @staticmethod
    def print_memory_usage(rss = True, python = True):
        if rss:
            MemUtil.print_rss_memory_usage()
        if python:           
            MemUtil.print_python_memory_usage()