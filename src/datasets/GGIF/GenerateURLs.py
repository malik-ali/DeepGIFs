#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import sys, os
import time, random, math
import argparse

QUERY_LIMIT = 500
RATING = 'g'
PAUSE_TIME = 2.0

from Constants import *

def query_giphy (search_term, n, offset):
    params = { 'api_key' : API_KEY, 'q' : search_term, 'limit' : QUERY_LIMIT, 'offset' : offset, 'rating' : RATING }
    response = requests.get (SEARCH_ENDPOINT, params=params)
    if response.status_code != requests.codes.ok:
        print (f'Bad response code ({response.status_code})... pausing for f{PAUSE_TIME} seconds.')
        time.sleep (PAUSE_TIME)
        return None

    urls = [ datum['images']['downsized']['url'] for datum in response.json ()['data'] ]
    return urls

def get_urls (query, n):
    print (f'Querying {SEARCH_ENDPOINT} for {n} \'{query}\' gifs... ', end="")
    offset = 0
    urls = []

    while len (urls) < n:
        result = query_giphy (query, n, offset)
        if len(result) == 0:
            break
        if not result:
            continue
        
        urls    += result
        offset  += QUERY_LIMIT

    print ('Done.')
    return urls

def save_urls (urls, out_path):
    if not os.path.exists (out_path):
        os.makedirs (out_path)

    print (f'Shuffling URLs...')
    random.shuffle (urls)

    print (f'Splitting {len(urls)} URLs:')
    cumulative = 0.0
    for split, fraction in SPLITS.items():
        out_file = os.path.join (out_path, f'{split}.txt')
        if os.path.exists (out_file):
            print (f' * Removing {out_file}')
            os.remove (out_file)

        split_start = math.floor (len (urls) * (cumulative))
        split_end   = math.floor (len (urls) * (cumulative + fraction))
        split_urls  = urls[split_start:split_end]
        cumulative += fraction

        print (f' * Saving {len(split_urls)} to {out_file}')
        with open (out_file, 'w') as f:
            f.writelines ([ url.strip () + '\n' for url in split_urls if url ])

def main ():
    parser = argparse.ArgumentParser (description='Find Giphy URLs for the provided search term.')
    parser.add_argument ('query', type=str)
    parser.add_argument ('number', type=int)
    
    args = parser.parse_args ()
    query, n = args.query, args.number
    out_path = os.path.join (URL_BASE, query)

    urls = get_urls (query, n)
    save_urls (urls[:n], out_path)
    print ('Done.')

if __name__ == '__main__':
    main ()
