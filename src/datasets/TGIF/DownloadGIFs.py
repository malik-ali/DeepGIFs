#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import urllib
import urllib.request
import tempfile
import threading as thr
from os.path import join

import multiprocessing

NUM_THREADS=16

def get_filelist(filelist_url):
    lines = urllib.request.urlopen(filelist_url)
    video_filenames = []
    for line in lines:
        line = line.decode('utf-8')
        video_filename = line.rstrip('\n')
        video_filenames.append(video_filename)
    return video_filenames

def download_files(urls, output_path, thread_num=1):
    START_TRIM = len('https://')
    
    os.makedirs(output_path, exist_ok=True)
    num_urls = len(urls)
    
    for i, url in enumerate(urls):
        filename = url[START_TRIM:].replace('/', '.')
        if i % 10 == 0:
            print(f"Thread {thread_num} - Downloading: {i}/{num_urls}")
        download_file(url, os.path.join(output_path, filename))
            
    print(f"Thread {thread_num} - finished")

def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isfile(out_file):
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        urllib.request.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)

def main():
    prefixes = ['train', 'val', 'test']
    for prefix in prefixes:
        with open(f'./URLs/{prefix}.txt', 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            output_path = f'./GIFs/{prefix}'
            multi_download_files(lines, output_path)

            

def multi_download_files(url_list, output_path):
    print("Multithreaded download...")
    
    llen = len(url_list) // NUM_THREADS
    split_names = [url_list[i*llen: (i + 1)* llen] for i in range(NUM_THREADS + 1)] 

    threads = []
    for i, urls in enumerate(split_names):
        th = thr.Thread(
            target=download_files, 
            args=(urls,), 
            kwargs={'output_path' : output_path, 'thread_num': i + 1}, 
            daemon=True)
        threads.append(th)
        th.start()

    for th in threads:
        th.join()

if __name__ == "__main__":
    main()
