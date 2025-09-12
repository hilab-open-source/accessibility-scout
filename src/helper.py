# https://medium.com/@fm194210/how-to-measure-the-execution-time-of-a-python-function-with-decorators-c030c3202064
def timeit(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"time taken by {func.__name__} is {time.time()-start }")

        return result

    return wrapper
