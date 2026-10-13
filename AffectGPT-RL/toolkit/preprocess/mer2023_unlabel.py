import os
import cv2
import tqdm
import glob
import numpy as np

# find error frames.npy
def analyze_frame_numbers(face_root, save_path):
    name2len = {}
    for face_dir in tqdm.tqdm(glob.glob(face_root + '/*')):
        facename = os.path.basename(face_dir)
        face_npy  = os.path.join(face_dir, facename + '.npy')
        try:
            faces = np.load(face_npy) # empty frame load cause errors
            if len(faces) <= 16: print (facename)
            name2len[facename] = len(faces)
        except:
            print (f'error file: {facename}')
    np.savez_compressed(save_path, 
                        name2len=name2len)

def analyze_name2len(face_root, store_path):
    name2len = np.load(store_path, allow_pickle=True)['name2len'].tolist()
    lens = [name2len[name] for name in name2len]
    print (f'sample number: {len(name2len)}') # 73981
    print (f'min len: {np.min(lens)}')   # 0
    print (f'max len: {np.max(lens)}')   # 885
    print (f'mean len: {np.mean(lens)}') # 85

    # 找 len = 0 的 name 并删除这些人脸
    for name in name2len:
        if name2len[name] <= 16:
            print (f'less than 16 faces: {name}')
            face_dir = os.path.join(face_root, name)
            if os.path.exists(face_dir):
                os.system(f'rm -rf {face_dir}')

if __name__ == '__main__':
    ## 直接用ghelper客户端 => clash and ghelper 的端口号是不一样的
    # set http_proxy=http://127.0.0.1:9981
    # set https_proxy=http://127.0.0.1:9981
    data_root = '/share/home/lianzheng/chinese-mer-2023/dataset/mer2023-dataset-unlabel'
    face_root = os.path.join(data_root, 'openface_face')
    save_path = os.path.join(data_root, 'unlabel-name2len.npz')
    # analyze_frame_numbers(face_root, save_path)
    # analyze_name2len(face_root, save_path) 
    # -> 最终剩下 73953 samples 用于预训练
