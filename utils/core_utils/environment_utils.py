import os
from .decorators import log_info


@log_info(print_enabled=True)
def clear_logs(log_directory: str, clear_prefix: str = "log") -> None:
    for filename in os.listdir(log_directory):
        filepath = f"{log_directory}/{filename}"
        if (filename.split("_")[0] == clear_prefix) and (filename.endswith("")):
            # Clear the file contents by opening in write mode and closing:
            print(f"    Resetting log: {filepath}")
            open(filepath, "w").close()
    print(f"WARNING: Cleared logs with prefix '{clear_prefix}'.")
    return
