import time


def timed(func):
    start = time.time()
    result = func()
    return time.time() - start, result
