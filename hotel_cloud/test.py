import os
import pandas as pd
import argparse

cli_parser = argparse.ArgumentParser(description='List the content of a folder')


cli_parser.add_argument('Path',
                       metavar='path',
                       type=str,
                       default="test_path", 
                       help='the path to list')

args = cli_parser.parse_args()


print(args)
print(args.Path)
