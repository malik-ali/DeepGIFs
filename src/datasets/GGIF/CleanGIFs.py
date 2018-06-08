#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from PIL import Image, ImageSequence, ImageFile
from Constants import *

PRINT_EVERY = 100

def clean_in_path (path):
    print (f'Cleaning path {path}...')
    
    entries = os.listdir (path)
    for index, dirent in enumerate (entries):
        gif_path = os.path.join (path, dirent)
        gif = Image.open (gif_path)
        
        try:
            frames = [ frame.copy () for frame in ImageSequence.Iterator (gif) ]
        except ValueError:
            print (f' * Removing {gif_path} (bad GIF metadata)')
            os.remove (gif_path)
        except IOError:
            print (f' * Removing {gif_path} (truncated GIF)')
            os.remove (gif_path)
            
        if index % PRINT_EVERY == 0:
            print (f' * Cleaned {index}/{len (entries)}')

    print ('Done.')

def main ():
    parser = argparse.ArgumentParser (description='Clean data for a certain search term.')
    parser.add_argument ('query', type=str)
    args = parser.parse_args ()
    
    gif_query_path = os.path.join (GIF_BASE, args.query)
    if not os.path.exists (gif_query_path):
        raise RuntimeError (f'No path {url_query_path}; try running GenerateURLS.py {args.query} <n>')
    
    for prefix in PREFIXES:
        gif_path = os.path.join (gif_query_path, prefix)
        if not os.path.exists (gif_path):
            raise RuntimeError (f'No path {gif_path}; try deleting {gif_query_path} and running GenerateURLs.py {args.query} <n>')
        clean_in_path (gif_path)

if __name__ == '__main__':
    main ()
