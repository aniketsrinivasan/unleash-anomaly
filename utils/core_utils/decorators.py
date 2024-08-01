import functools
import time


def log_info(log_path: str = None, log_enabled: bool = False, print_enabled: bool = True):
    def log_info_decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            now = time.time()
            string = (f"{now}: Called function {function.__qualname__} with following information:\n"
                      f"   args:   {[str(arg) for arg in args]}\n"
                      f"   kwargs: {[f'{str(kwarg)} = {str(kwargs[kwarg])}' for kwarg in kwargs]}")
            # Log if log_enabled is True and log_path is provided:
            if (log_enabled is True) and (log_path is not None):
                with open(log_path, "a") as log_file:
                    log_file.write(string)
                    log_file.write("\n\n")
            # If log_enabled is True but no log_path is provided:
            elif (log_enabled is True) and (log_path is None):
                print(f"{now}: DecoratorWarn: @log_info used in function {function.__qualname__} but a "
                      f"logging path was not provided. Not logging information saved.")
            # If printing is turned on:
            if print_enabled:
                print(string)
            return function(*args, **kwargs)
        return wrapper
    return log_info_decorator
