import sys
import os
import urllib.request
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd

import ftfy
import html
import regex as re

data_folder = 'data'
os.makedirs(data_folder, exist_ok=True)

dataset_file = 'dataset.tsv'
root_url = "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images"
error_file = 'file_error.log'
errors = 0


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'\%..', '', text)
    return text

def get_file(sample):
    global errors
    h, p, d = sample['hash'], str(sample['pid']), sample['description']
    download_url = f'{root_url}/{h[0:3]}/{h[3:6]}/' + h + '.jpg'
    download_path = os.path.join(data_folder, p + '.jpg')
    if not os.path.isfile(download_path):
        try:
            urllib.request.urlretrieve(download_url, download_path)
            with open(dataset_file, 'a') as f:
                f.write(f'{p}\t{d}\t{download_url}\n')
        except Exception as e:
            with open(error_file, "a") as f:
                f.write("{} - {}\n".format(download_url, e))
                errors += 1
    else:                  
        with open(dataset_file, 'a') as f:
            f.write(f'{p}\t{d}\t{download_url}\n')

if __name__ == '__main__':
    with open(dataset_file, 'w') as f:
        f.write('pid\tdescription\turl\n')

    df = pd.read_csv('yfcc100m_subset_data.tsv', '\t').values.tolist()
    df_dict = {int(d[0]): [d[1], d[2]] for d in df}
    n_lines = 0
    n_files = 0
    with open('new_metadata.csv') as f:
        f.readline()
        for line in f:
            if int(n_lines) in df_dict:
                get_file(
                    {
                        'pid': df_dict[n_lines][0],
                        'description': basic_clean(line.strip().split('\t')[7]),
                        'hash': df_dict[n_lines][1]
                    }
                )
                print(f'[{n_files+1}/{n_lines+1}/{len(df_dict)}]')
                n_files += 1
            n_lines += 1



