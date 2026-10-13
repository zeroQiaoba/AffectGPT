import os
import cv2
import glob
import tqdm
import pickle
import multiprocessing
from toolkit.utils.chatgpt import *
from toolkit.utils.functions import *
from toolkit.utils.read_files import *
from toolkit.globals import *

# t: ms
def convert_time(t):
    t = int(t)
    ms = t % 1000
    t = math.floor(t / 1000)
    h = math.floor(t / 3600)
    m = math.floor((t - h * 3600) / 60)
    s = t - 3600 * h - 60 * m
    return '%02d:%02d:%02d.%03d' % (h, m, s, ms)


def split_video_by_start_end_IEMOCAP(data_root, save_root1, save_root2):
    ffmpeg_path = config.PATH_TO_FFMPEG
    if not os.path.exists(save_root1): os.makedirs(save_root1)
    if not os.path.exists(save_root2): os.makedirs(save_root2)

    error_lines = []
    for session_name in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
        avi_root = os.path.join(data_root, session_name, 'dialog/avi/DivX')
        transcription_root = os.path.join(data_root, session_name, 'dialog/transcriptions')

        for trans_path in glob.glob(transcription_root+'/S*.txt'):
            #trans_path = f'{data_root}/Session1/dialog/transcriptions/Ses01F_impro01.txt'
            trans_name = os.path.basename(trans_path)[:-4]
            avi_path = os.path.join(avi_root, trans_name + '.avi')
            mp4_path = os.path.join(save_root1, trans_name + '.mp4')

            ## change avi to mp4
            cmd = f'{ffmpeg_path} -i "{avi_path}" "{mp4_path}"'
            os.system(cmd)

            ## read lines
            with open(trans_path, encoding='utf8') as f: lines = [line.strip() for line in f]
            lines = [line for line in lines if len(line)!=0]
            for line in lines: # line: Ses05F_script03_1_F033 [241.6700-243.4048]: You knew there was nothing.
                try: # some line cannot be processed
                    subname = line.split(' [')[0]
                    subvideo_path = os.path.join(save_root2, subname+'.mp4')

                    start = line.split('[')[1].split('-')[0]
                    end = line.split('-')[1].split(']')[0]
                    start = convert_time(float(start)*1000)
                    end = convert_time(float(end)*1000)

                    cmd = f'{ffmpeg_path} -ss {start} -to {end} -accurate_seek -i "{mp4_path}" -vcodec copy -acodec copy "{subvideo_path}" -y'
                    os.system(cmd)
                except:
                    error_lines.append(line)
                    continue
    print (error_lines)


def generate_transcription_files_IEMOCAP(data_root, csv_file):
    names = []
    sentences = []
    for session_name in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
        transcription_root = os.path.join(data_root, session_name, 'dialog/transcriptions')
        for trans_path in glob.glob(transcription_root+'/S*.txt'):
            with open(trans_path, encoding='utf8') as f: lines = [line.strip() for line in f]
            lines = [line for line in lines if len(line)!=0]
            for line in lines: # line: Ses05F_script03_1_F033 [241.6700-243.4048]: You knew there was nothing.
                try: # some line cannot be processed
                    subname = line.split(' [')[0]
                    start = line.split('[')[1].split('-')[0]
                    end = line.split('-')[1].split(']')[0]
                    start = convert_time(float(start)*1000)
                    end = convert_time(float(end)*1000)
                    sentence = line.split(']:')[1].strip()
                    names.append(subname)
                    sentences.append(sentence)
                except:
                    continue

    ## write to csv file
    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [sentences[ii]]
    func_write_key_to_csv(csv_file, names, name2key, ['english'])


def label_convertion(label_pkl, save_path):
    names, labels = [], []
    videoIDs, videoLabels, _, _, trainVids, testVids = pickle.load(open(label_pkl, "rb"), encoding='latin1')
    vids = sorted(list(trainVids | testVids))
    for vid in vids:
        names.extend(videoIDs[vid])
        labels.extend(videoLabels[vid])
    print (f'sample number: {len(names)}')

    whole_corpus = {}     
    for ii, name in enumerate(names):
        whole_corpus[name] = {'emo': labels[ii], 'val': -10}
            
    np.savez_compressed(save_path,
                        whole_corpus=whole_corpus)

# ----------------------------------------------------------- #
def func_find_tgt_pos(videoname):
    left_gender = videoname[5] # 'M' or 'F'
    target_gender = videoname[-4] # 'M' or 'F'
    assert left_gender in ['M', 'F']
    assert target_gender in ['M', 'F']
    if left_gender == target_gender:
        return 'left'
    else:
        return 'right'

def func_crop_tgt_frame(frame, part='left'):
    h, w, _ = frame.shape
    if part == 'left':
        frame = frame[:, :int(w/2), :]
    else:
        frame = frame[:, int(w/2):, :]
    return frame

def convert_for_one_video(argv=None, video_path=None, save_path=None):
    video_path, save_path = argv
    
    # 找到目标人在的位置
    videoname = os.path.basename(video_path).rsplit('.', 1)[0]
    tgt_pos = func_find_tgt_pos(videoname)

    # 存储所有 frames
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while 1:
        ## read frame and detect face
        has_frame, frame = cap.read()
        if not has_frame:
            break
        ## crop frame
        frame = func_crop_tgt_frame(frame, tgt_pos)
        frames.append(frame)
    cap.release()
    
    # 数据存储到.avi
    height, width, _ = frames[0].shape
    videoWriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (int(width), int(height)), True)
    for frame in frames:
        videoWriter.write(frame)
    videoWriter.release()

def convert_for_all_video_multiprocess(video_root, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)

    params = []
    for video_path in glob.glob(video_root + '/*'):
        video_name = os.path.basename(video_path)[:-4]
        save_path = os.path.join(save_root, video_name+'.avi')
        params.append((video_path, save_path))

    with multiprocessing.Pool(processes=8) as pool:
        list(tqdm.tqdm(pool.imap(convert_for_one_video, params), total=len(params)))

# ----------------------------------------------------------- #
def normalize_dataset_format(data_root, save_root):
    
    ## generate subvideo
    save_root1 = os.path.join(save_root, 'video')
    save_root2 = os.path.join(save_root, 'subvideo')
    split_video_by_start_end_IEMOCAP(data_root, save_root1, save_root2)

    ## generate transcripts
    csv_file = os.path.join(save_root, 'transcription.csv')
    generate_transcription_files_IEMOCAP(data_root, csv_file)

    ## label conversion (4-way | 6-way)
    label_4way_old = os.path.join(save_root, 'IEMOCAP_features_raw_4way.pkl')
    label_4way_new = os.path.join(save_root, 'label_4way.npz')
    assert os.path.exists(label_4way_old), 'label file should exist'
    label_convertion(label_4way_old, label_4way_new)

    label_6way_old = os.path.join(save_root, 'IEMOCAP_features_raw_6way.pkl')
    label_6way_new = os.path.join(save_root, 'label_6way.npz')
    assert os.path.exists(label_6way_old), 'label file should exist'
    label_convertion(label_6way_old, label_6way_new)

    ## gain target speaker face
    video_root = os.path.join(save_root, 'subvideo')
    save_root  = os.path.join(save_root, 'subvideo-tgt')
    convert_for_all_video_multiprocess(video_root, save_root)
    

if __name__ == '__main__':
    ## linux
    # data_root = '/share/home/lianzheng/emotion-data/IEMOCAP_full_release'
    # save_root = '/share/home/lianzheng/chinese-mer-2023/dataset/iemocap-process'
    # normalize_dataset_format(data_root, save_root)

    ## window => 直接用ghelper客户端 => clash and ghelper 的端口号是不一样的 [linux无法翻墙]
    # set http_proxy=http://127.0.0.1:9981
    # set https_proxy=http://127.0.0.1:9981
    # translate transcript
    # data_root = 'H:\\desktop\\Multimedia-Transformer\\chinese-mer-2023\\dataset\\iemocap-process'
    # trans_path = os.path.join(data_root, 'transcription.csv')
    # save_path  = os.path.join(data_root, 'transcription-engchi.csv')
    # polish_path = os.path.join(data_root, 'transcription-engchi-polish.csv')
    # # func_translate_transcript(trans_path, save_path)  # 还是需要人为检查一下，尤其是带“翻译”的单词
    # func_translate_transcript_polish(trans_path, save_path, polish_path) # 再次检测一下遗漏的部分

    ## check for label mapping
    # label_path = config.PATH_TO_LABEL['IEMOCAPFour']
    # corpus = np.load(label_path, allow_pickle=True)['whole_corpus'].tolist()
    # labels = [corpus[name]['emo'] for name in corpus]
    # func_discrte_label_distribution(labels)
    '''
    [happy, sad, neutral, anger]
    [1636, 1084, 1708, 1103]
    0 :  1636
    1 :  1084
    2 :  1708
    3 :  1103
    '''
    label_path = config.PATH_TO_LABEL['IEMOCAPSix']
    corpus = np.load(label_path, allow_pickle=True)['whole_corpus'].tolist()
    labels = [corpus[name]['emo'] for name in corpus]
    func_discrte_label_distribution(labels)
    '''
    [happy, sad, neutral, angry, excited, frustrate]
    [648, 1084, 1708, 1103, 1041, 1849]
    0 :  648
    1 :  1084
    2 :  1708
    3 :  1103
    4 :  1041
    5 :  1849
    '''
    