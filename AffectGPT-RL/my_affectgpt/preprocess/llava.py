import tqdm
import config
from toolkit.utils.read_files import *
from toolkit.utils.functions import *
from my_affectgpt.evaluation.ew_metric import *

dataset = 'LLaVA'

##################################################
## keep only the emotion-related caption information
##################################################
def convert_json_to_csv(image_root, label_path, save_path):

    # read infos
    store_names, store_qas = [], []
    with open(label_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    for data in tqdm.tqdm(datas):
        name = data['image']
        qas  = data['conversations']

        imageopath = os.path.join(image_root, name)
        assert os.path.exists(imageopath)
        
        if len(qas) == 2:
            question = qas[0]['value']
            answer = qas[1]['value']
            question = question.replace('<Image>', '').replace('<image>', '')
            question = question.replace('\n', ' ').replace('\t', ' ').strip()
            answer   = answer.replace('\n', ' ').replace('\t', ' ').strip()
            store_names.append(name)
            store_qas.append([question, answer])
    print (f'sample number: {len(store_names)}') # 100547 (name, q, a)
    # keep the non-conversation data only: 158k -> 100k
    
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
    print (f'sample number: {len(nq2a)}') # 100547 [ok]
    
    # # step2 filter caption: recognize sentiment from the caption
    # llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')
    # # 1. get openset
    # nq2openset = {}
    # whole_nqs, whole_opensets = extract_openset_batchcalling(name2reason=nq2a, 
    #                                                          llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
    # for nq, openset in zip(whole_nqs, whole_opensets):
    #     nq2openset[nq] = openset
    # # 2. save
    # save_temp = os.path.join(config.DATA_DIR[dataset], 'nq2openset.npy')
    # np.save(save_temp, nq2openset)
        
    ## step3: statistics
    save_temp = os.path.join(config.DATA_DIR[dataset], 'nq2openset.npy')
    nq2openset = np.load(save_temp, allow_pickle=True).tolist()
    print (f'sample number: {len(nq2openset)}') # 100547
    emo_nqs = [nq for nq in nq2openset if string_to_list(nq2openset[nq]) != []]
    print (f'emotion sample number: {len(emo_nqs)}') # 35671

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

## step2: 
'''
100k not-conversation data -> 35k emotion-related data
'''


def remove_not_used_images(label_path, image_root):

    # number of videos present in the labels
    names = func_read_key_from_csv(label_path, 'name')
    print ('labeled sample number: ', len(set(names))) # 78886

    # number of videos present in the path
    image_paths = []
    for image_path in tqdm.tqdm(glob.glob(image_root + '/*')):
        image_paths.append(image_path)
    print (f'image sample in root number: {len(image_paths)}') # 118287

    # delete files
    for image_path in tqdm.tqdm(glob.glob(image_root + '/*')):
        image_name = os.path.basename(image_path)
        if image_name not in names:
            os.remove(image_path)


# step3: delete videos not used anywhere => [this is the already-filtered file]


def ensure_all_image_exist(label_path, image_root):

    # number of videos present in the labels
    names = func_read_key_from_csv(label_path, 'name')
    print ('labeled sample number: ', len(set(names))) #

    # number of videos present in the path
    for name in tqdm.tqdm(names):
        image_path = os.path.join(image_root, name)
        assert os.path.exists(image_path)
        

# step4: make sure every image still exists after step3
