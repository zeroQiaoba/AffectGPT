import tqdm
import config
from toolkit.utils.read_files import *
from toolkit.utils.functions import *
from my_affectgpt.evaluation.ew_metric import *

dataset = 'EmoVIT'

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

    print (f'qa number: {len(store_names)}') # 54326 (name, q, a)
    print (f'sample number: {len(set(store_names))}') # 42400 unique samples

    # store into save_path
    columns = ['name', 'question', 'answer']
    data = np.column_stack([store_names, store_qas])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)


## Step1: label conversion [drop the conversation part so questions do not depend on context]


def preserve_not_direct_indicate_emotion_parts(label_path, save_path):

    ## step1: read label_path
    names = func_read_key_from_csv(label_path, 'name')
    questions = func_read_key_from_csv(label_path, 'question')
    answers = func_read_key_from_csv(label_path, 'answer')

    names_filter, questions_filter, answers_filter = [], [], []
    for (name, question, answer) in zip(names, questions, answers):
        if question.find("Please select the emotion closest to the image from the following options") == -1:
            names_filter.append(name)
            questions_filter.append(question)
            answers_filter.append(answer)
    print (f'filtered sample number: {len(names_filter)}') # 17526
    
    ## step4: result storage
    columns = ['name', 'question', 'answer']
    data = np.column_stack([names_filter, questions_filter, answers_filter])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)

## step2: 
# label_path = os.path.join(config.DATA_DIR[dataset], 'label.csv')
# save_path  = os.path.join(config.DATA_DIR[dataset], 'label-remove-direct-emotion.csv')
# preserve_not_direct_indicate_emotion_parts(label_path, save_path)


def remove_not_used_images(label_path, image_root):

    # number of videos present in the labels
    names = func_read_key_from_csv(label_path, 'name')
    print ('labeled sample number: ', len(set(names))) # 42400

    # number of videos present in the path
    image_paths = []
    for image_path in tqdm.tqdm(glob.glob(image_root + '/*')):
        image_paths.append(image_path)
    print (f'image sample in root number: {len(image_paths)}') # 118102

    # delete files
    for image_path in tqdm.tqdm(glob.glob(image_root + '/*')):
        image_name = os.path.basename(image_path)
        if image_name not in names:
            os.remove(image_path)


# step3: delete videos not used anywhere => [this is the already-filtered file]



def extract_ov_labels(label_path, save_path):
    names = func_read_key_from_csv(label_path, 'name')
    questions = func_read_key_from_csv(label_path, 'question')
    answers = func_read_key_from_csv(label_path, 'answer')
    nq2answer = {}
    for (name, question, answer) in zip(names, questions, answers):
        nq2answer[name + '+' + question] = answer
    print (f'sample number: {len(nq2answer)}') # 54326 samples

    # nq2answer => openset
    llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')
    extract_openset_batchcalling(name2reason=nq2answer, store_npz=save_path,
                                 llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)

def calculate_statistic_for_ov(save_path):
    fileitems = np.load(save_path, allow_pickle=True)['fileitems']
    print (f'sample number: {len(fileitems)}') # 54326

    whole_ovs = []
    for item in fileitems:
        whole_ovs.extend(string_to_list(item))
    print (set(whole_ovs))
    print (f'unique label: {len(set(whole_ovs))}') # 1074

    # count labels appearing more than 6 times
    label_filter = []
    label2count = func_label_distribution(whole_ovs)
    for label in label2count:
        if label2count[label] >= 6:
            label_filter.append(label)
    print (label_filter)
    print (f'at least appear 6 times: {len(set(label_filter))}') # 282
    

## step4: extract ov labels from label.csv
label_path = os.path.join(config.DATA_DIR[dataset], 'label.csv')
save_path = label_path[:-4]+'-ov.npz'
## step5: count labels
calculate_statistic_for_ov(save_path)
