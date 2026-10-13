import tqdm
import config
from toolkit.utils.read_files import *
from toolkit.utils.functions import *
from my_affectgpt.evaluation.ew_metric import *

dataset = 'LLaVAVideo'

##################################################
## keep only the emotion-related caption information
##################################################
def all_english(text):
    # [A-Za-z0-9.,!?;:'"()\-] matches letters, digits and common punctuation
    pattern = re.compile(r'[A-Za-z0-9.,!?;:\'"()\- ]')
    return all(pattern.match(char) or char.isspace() for char in text)


def convert_json_to_csv(save_path):

    # get label
    store_names, store_qas = [], []
    data_root = config.DATA_DIR[dataset]
    for file_root in glob.glob(data_root + '/0_30_s*'):
        for file_path in glob.glob(file_root + '/*.json'):
            # skip the oe, cap and mc parts: their format differs from normal captions and needs extra handling
            if file_path.find('_oe_') != -1 or file_path.find('_cap_') != -1:
                print (file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    datas = json.load(f)
                for data in tqdm.tqdm(datas):
                    name = data['video']
                    conversations = data['conversations']
                    assert len(conversations) % 2 == 0
                    for ii in range(int(len(conversations) / 2)):
                        question = conversations[ii*2]['value']
                        answer   = conversations[ii*2+1]['value']
                        question = question.replace('<image>', ' ').replace('\n', ' ').replace('\t', ' ').strip()
                        answer   = answer.replace('<image>', ' ').replace('\n', ' ').replace('\t', ' ').strip()
                        ## three extra constraints: all english; video format; non-empty question and answer
                        if not all_english(question) or not all_english(answer): continue
                        if not name.endswith('.mkv') and not name.endswith('.mp4'): continue
                        if len(question) == 0 or len(answer) == 0: continue
                        store_names.append(name)
                        store_qas.append([question, answer])
    print (f'sample number: {len(store_names)}') # 1,962,654
    print (f'unique sample number: {len(set(store_names))}') # 86,717

    # judge whether video exists
    error_videos = []
    video_root = config.PATH_TO_RAW_VIDEO[dataset]
    for name in set(store_names):
        video_path = os.path.join(video_root, name)
        try:
            assert os.path.exists(video_path)
        except:
            error_videos.append(name)
    print (f'Does not exist number: {len(error_videos)}') # 1,018

    # filter samples
    filter_names, filter_qas = [], []
    for (name, [q, a]) in zip(store_names, store_qas):
        if name not in error_videos:
            filter_names.append(name)
            filter_qas.append([q, a])

    # store into save_path
    columns = ['name', 'question', 'answer']
    data = np.column_stack([filter_names, filter_qas])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)

## Step1: label conversion [drop the conversation part so questions do not depend on context]



###################################################
## find errors in the data processing
from decord import VideoReader
from my_affectgpt.processors.video_processor import load_video
def test_video_readable(label_path, video_root):
    ## 1. find all unique names
    
    ## 2. find the error source by unique name => the videos themselves are the problem

    ## 3. can basename alone identify the video? => no
    names = np.load('names.npy')
    basenames = [os.path.basename(name) for name in names]
    print (len(set(names))) # 85k
    print (len(set(basenames))) # 78k
    



#####################################################
## post-process the data for the errors found
def func_video_conversion_multiprocess(argv=None, video_path=None, save_path=None):
    video_path, save_path = argv
    cmd = "%s -loglevel quiet -y -i \"%s\" \"%s\"" %(config.PATH_TO_FFMPEG, video_path, save_path)
    os.system(cmd)


def main_video_conversion_multiprocess(video_root, save_root):
    params = []
    names = np.load('names.npy')
    for name in names:
        raw_video = os.path.join(video_root, name)
        assert os.path.exists(raw_video)

        save_video = os.path.join(save_root, name)
        temp_root = os.path.split(save_video)[0]
        if not os.path.exists(temp_root): 
            os.makedirs(temp_root)
        params.append((raw_video, save_video))

        # print(raw_video)
        # print(save_video)

    with multiprocessing.Pool(processes=8) as pool:
        list(tqdm.tqdm(pool.imap(func_video_conversion_multiprocess, params), total=len(params)))

# video_root = os.path.join(config.DATA_DIR[dataset], 'video')
# save_root  = os.path.join(config.DATA_DIR[dataset], 'video-new')
# main_video_conversion_multiprocess(video_root, save_root)



#####################################################
## keep only emotion-related captions
def perseve_emotion_related_description(label_path, save_path):

    # step1: read label_path
    names = func_read_key_from_csv(label_path, 'name')
    questions = func_read_key_from_csv(label_path, 'question')
    answers = func_read_key_from_csv(label_path, 'answer')
    nq2a = {}
    for (name, question, answer) in zip(names, questions, answers):
        nq = f'{name}+{question}'
        nq2a[nq] = answer
    print (f'sample number: {len(nq2a)}') # 1,962,654
    
    # step2 filter caption: recognize sentiment from the caption
    llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')
    # 1. get openset
    nq2openset = {}
    whole_nqs, whole_opensets = extract_openset_batchcalling(name2reason=nq2a, 
                                                             llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
    for nq, openset in zip(whole_nqs, whole_opensets):
        nq2openset[nq] = openset
    # 2. save
    save_temp = os.path.join(config.DATA_DIR[dataset], 'nq2openset.npy')
    np.save(save_temp, nq2openset)


label_path = os.path.join(config.DATA_DIR[dataset], 'label.csv') # 515k
save_path  = os.path.join(config.DATA_DIR[dataset], 'label-onlyemotion.csv')
perseve_emotion_related_description(label_path, save_path)




## multi-process audio extraction
def func_split_audio_multiprocess(argv=None, video_path=None, audio_path=None):
    video_path, audio_path = argv
    cmd = "%s -loglevel quiet -y -i %s -ar 16000 -ac 1 %s" %(config.PATH_TO_FFMPEG, video_path, audio_path)
    os.system(cmd)

def main_split_audio_multiprocess(label_path, video_root, save_root):
    
    names = func_read_key_from_csv(label_path, 'name')
    video_paths = [os.path.join(video_root, name)for name in names]

    params = []
    for video_path in tqdm.tqdm(video_paths):
        videoname = video_path[:-4] + '.wav'
        audio_path = os.path.join(save_root, videoname)
        audio_root = os.path.split(audio_path)[0]
        if not os.path.exists(audio_root): 
            os.makedirs(audio_root)
        params.append((video_path, audio_path))
    
    with multiprocessing.Pool(processes=8) as pool:
        list(tqdm.tqdm(pool.imap(func_split_audio_multiprocess, params), total=len(params)))


## single-process audio extraction
def main_split_audio(label_path, video_root, save_root):
    
    names = func_read_key_from_csv(label_path, 'name')
    video_paths = [os.path.join(video_root, name)for name in names]

    for video_path in tqdm.tqdm(video_paths):
        videoname = video_path[:-4] + '.wav'
        audio_path = os.path.join(save_root, videoname)
        audio_root = os.path.split(audio_path)[0]
        if not os.path.exists(audio_root): 
            os.makedirs(audio_root)

        cmd = "%s -loglevel quiet -y -i %s -ar 16000 -ac 1 %s" %(config.PATH_TO_FFMPEG, video_path, audio_path)
        os.system(cmd)
