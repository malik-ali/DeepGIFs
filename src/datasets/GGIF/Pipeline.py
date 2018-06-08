#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess

def main ():
    parser = argparse.ArgumentParser (description='Execute GenerateURLs.py, DownloadGIFs.py, and CleanGIFs.py in sequence.')
    parser.add_argument ('query', type=str)
    parser.add_argument ('number', type=int)
    args = parser.parse_args ()
    
    q = args.query
    n = args.number

    print (f'Pipeline: ./GenerateURLs.py {q} {n}')
    subprocess.Popen (f'./GenerateURLs.py {q} {n}', shell=True).wait ()
    print ()

    print (f'Pipeline: ./DownloadGIFs.py {q}')
    subprocess.Popen (f'./DownloadGIFs.py {q}', shell=True).wait ()
    print ()

    print (f'Pipeline: ./CleanGIFs.py {q}')
    subprocess.Popen (f'./CleanGIFs.py {q}', shell=True).wait ()
    print ()

if __name__ == '__main__':
    main ()
