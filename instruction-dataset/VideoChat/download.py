import os
import argparse
import requests
import warnings 
import numpy as np
import pandas as pd
import concurrent.futures
from mpi4py import MPI # similar to multiprocess

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank() # RANK=0
SIZE = COMM.Get_size() # SIZE=1

# read target samples
import json
def read_names_from_json(json_path):
    whole_names = []
    with open(json_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        videopath = item['video']
        whole_names.append(videopath)
    return whole_names

# download url and save to 'save_fp=[save filepath]'
def request_save(url, save_fp):
    print (f'download {save_fp}')
    img_data = requests.get(url, timeout=5).content
    with open(save_fp, 'wb') as handler:
        handler.write(img_data)

def main(args):

    ### read target json
    whole_names = read_names_from_json(args.json_path)

    ### preproc
    video_dir = os.path.join(args.data_dir, 'videos')
    if RANK == 0:
        if not os.path.exists(os.path.join(video_dir, 'videos')):
            os.makedirs(os.path.join(video_dir, 'videos'))
    COMM.barrier()

    # ASSUMES THE CSV FILE HAS BEEN SPLIT INTO N PARTS
    partition_dir = args.csv_path.replace('.csv', f'_{args.partitions}') # results_2M_train_1

    # if not, then split in this job.
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)
        full_df = pd.read_csv(args.csv_path)
        df_split = np.array_split(full_df, args.partitions)
        for idx, subdf in enumerate(df_split):
            subdf.to_csv(os.path.join(partition_dir, f'{idx}.csv'), index=False)

    relevant_fp = os.path.join(args.data_dir, 'relevant_videos_exists.txt')
    if os.path.isfile(relevant_fp):
        exists = pd.read_csv(os.path.join(args.data_dir, 'relevant_videos_exists.txt'), names=['fn'])
    else:
        exists = []

    df = pd.read_csv(os.path.join(partition_dir, f'{args.part}.csv'))
    df['rel_fn'] = df.apply(lambda x: os.path.join(str(x['page_dir']), str(x['videoid'])), axis=1)
    df['rel_fn'] = df['rel_fn'] + '.mp4'
    df = df[~df['rel_fn'].isin(exists)]
    df.dropna(subset=['page_dir'], inplace=True) # remove nan
    playlists_to_dl = np.sort(df['page_dir'].unique())
    for page_dir in playlists_to_dl: # 3590
        vid_dir_t = os.path.join(video_dir, page_dir)
        pdf = df[df['page_dir'] == page_dir] # 4666
        if len(pdf) > 0:
            if not os.path.exists(vid_dir_t):
                os.makedirs(vid_dir_t)

            # process (urls_todo，save_fps)
            urls_todo = []
            save_fps = []
            for idx, row in pdf.iterrows():
                video_fp = os.path.join(vid_dir_t, str(row['videoid']) + '.mp4')
                ##################################
                ## skip not needed samples in VideoChat
                video_fp_name = '/'.join(video_fp.split('/')[-2:])
                if video_fp_name not in whole_names: continue
                ##################################
                if not os.path.isfile(video_fp):
                    urls_todo.append(row['contentUrl'])
                    save_fps.append(video_fp)

            print(f'Spawning {len(urls_todo)} jobs for page {page_dir}') # Spawning 4666 jobs for page 000001_000050
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.processes) as executor:
                future_to_url = {executor.submit(request_save, url, fp) for url, fp in zip(urls_todo, save_fps)}


# conda activate tf241env38
# python download.py --csv_path=results_2M_train.csv  --partitions=1 --part=0 --data_dir=./data --processes=8
# python download.py --csv_path=results_2M_val.csv    --partitions=1 --part=0 --data_dir=./data --processes=8
# python download.py --csv_path=results_10M_train.csv --partitions=1 --part=0 --data_dir=./data --processes=8
# python download.py --csv_path=results_10M_val.csv   --partitions=1 --part=0 --data_dir=./data --processes=8

# accerate: find target url and download
# python download.py --csv_path=results_10M_val.csv   --partitions=1 --part=0 --data_dir=./data --processes=8 --json_path='../Ask-Anything-main/dataset/videochat_instruct_11k.json'
# python download.py --csv_path=results_10M_train.csv --partitions=1 --part=0 --data_dir=./data --processes=8 --json_path='../Ask-Anything-main/dataset/videochat_instruct_11k.json'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shutter Image/Video Downloader')
    parser.add_argument('--partitions', type=int, default=4, help='Number of partitions to split the dataset into, to run multiple jobs in parallel')
    parser.add_argument('--part',       type=int, required=True, help='Partition number to download where 0 <= part < partitions')
    parser.add_argument('--data_dir',   type=str, default='./data', help='Directory where webvid data is stored.')
    parser.add_argument('--csv_path',   type=str, default='results_2M_train.csv', help='Path to csv data to download')
    parser.add_argument('--json_path',  type=str, default=None, help='Path to target json file to accelerate')
    parser.add_argument('--processes',  type=int, default=8)
    args = parser.parse_args()

    if SIZE > 1:
        warnings.warn("Overriding --part with MPI rank number")
        args.part = RANK

    if args.part >= args.partitions:
        raise ValueError("Part idx must be less than number of partitions")
    main(args)