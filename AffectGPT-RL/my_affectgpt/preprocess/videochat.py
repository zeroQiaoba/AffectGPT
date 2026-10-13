import tqdm
import config
from toolkit.utils.read_files import *
from toolkit.utils.functions import *
from my_affectgpt.evaluation.ew_metric import *

dataset = 'VideoChat'

##################################################
## keep only the emotion-related caption information
##################################################
def convert_json_to_csv(video_root, label_path, save_path):

    # read infos
    store_names, store_qas = [], []
    with open(label_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    for data in tqdm.tqdm(datas):
        name = data['video']
        qas  = data['QA']

        videopath = os.path.join(video_root, name)
        assert os.path.exists(videopath)
        
        if len(qas) == 1:
            qa = qas[0]
            question = qa['q'].replace('\n', ' ').replace('\t', ' ')
            answer   = qa['a'].replace('\n', ' ').replace('\t', ' ')
            store_names.append(name)
            store_qas.append([question, answer])
    print (f'sample number: {len(store_names)}') # 6636 (name, q, a)
    '''
    11k mix -> 7k caption
    '''
    
    # store into save_path
    columns = ['name', 'question', 'answer']
    data = np.column_stack([store_names, store_qas])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)


## Step1: label conversion [drop the conversation part so questions do not depend on context]



def perseve_emotion_related_description(label_path, save_path):

    ## step1: read label_path
    names = func_read_key_from_csv(label_path, 'name')
    questions = func_read_key_from_csv(label_path, 'question')
    answers = func_read_key_from_csv(label_path, 'answer')
    nq2a = {}
    for (name, question, answer) in zip(names, questions, answers):
        nq = f'{name}+{question}'
        nq2a[nq] = answer
    print (f'sample number: {len(nq2a)}') # 6636 [ok]
    
        
    ## step3: statistics
    save_temp = os.path.join(config.DATA_DIR[dataset], 'nq2openset.npy')
    nq2openset = np.load(save_temp, allow_pickle=True).tolist()
    print (f'sample number: {len(nq2openset)}') # 6636
    emo_nqs = [nq for nq in nq2openset if string_to_list(nq2openset[nq]) != []]
    print (f'emotion sample number: {len(emo_nqs)}') # 2993
    '''
    7k caption -> 3k emotion
    '''

    ## step4: result storage
    names_filter, questions_filter, answers_filter = [], [], []
    for nq in nq2a:
        if nq in emo_nqs:
            n, q = nq.split('+')
            a = nq2a[nq]
            names_filter.append(n)
            questions_filter.append(q)
            answers_filter.append(a)
    columns = ['name', 'question', 'answer']
    data = np.column_stack([names_filter, questions_filter, answers_filter])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)


## step3: 


def remove_not_used_videos(label_path, video_root):

    # number of videos present in the labels
    names = func_read_key_from_csv(label_path, 'name')
    questions = func_read_key_from_csv(label_path, 'q')
    answers = func_read_key_from_csv(label_path, 'a')
    print ('labeled sample number: ', len(set(names))) # 7905

    # number of videos present in the path
    video_paths = []
    for video_subroot in tqdm.tqdm(glob.glob(video_root + '/*')):
        for video_path in glob.glob(video_subroot + '/*'):
            video_paths.append(video_path)
    print (f'video sample in root number: {len(video_paths)}') # 7905

## step4: delete videos not used anywhere => [this is the already-filtered file]



'''
python affectgpt/preprocess/videochat.py
'''