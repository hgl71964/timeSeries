import os
import pandas as pd


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    INFO_CYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

a = {"a": 3}
print(f"{bcolors.HEADER}config: {bcolors.ENDC}")
print(f"{bcolors.FAIL}", a)