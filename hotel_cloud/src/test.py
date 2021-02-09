import os 
import argparse

cli = argparse.ArgumentParser(description="config global parameters")

cli.add_argument("--lb",
                dest="lb",
                type=int,
                nargs="+", 
                default=[1, 7, 14], 
                help="2 args -> lag range for lag feats")

args = cli.parse_args()

a = args.lb              # the bound for lagged features

print(a)