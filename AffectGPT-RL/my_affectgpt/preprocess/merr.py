import tqdm
import config
from toolkit.utils.read_files import *
from toolkit.utils.functions import *
from my_affectgpt.evaluation.ew_metric import *

# MERRCoarse / MERRFine are not part of this release, so their dataset modules are gone
# from my_affectgpt.datasets.datasets.merrcoarse_dataset import MERRCoarse_Dataset
# from my_affectgpt.datasets.datasets.merrfine_dataset import MERRFine_Dataset


######################################
## reason -> ov labels
######################################



######################################
## reason -> statistics
######################################
def calculate_statistic_for_ov(save_path):
    fileitems = np.load(save_path, allow_pickle=True)['fileitems']
    print (f'sample number: {len(fileitems)}') # 54326

    whole_ovs = []
    for item in fileitems:
        whole_ovs.extend(string_to_list(item))
    print (f'unique label: {len(set(whole_ovs))}') # 1074

    # count labels appearing more than 6 times
    label_filter = []
    label2count = func_label_distribution(whole_ovs)
    for label in label2count:
        if label2count[label] >= 6:
            label_filter.append(label)
    print (f'at least appear 6 times: {len(set(label_filter))}') # 282


dataset = 'MERRFine'
save_path = config.PATH_TO_LABEL[dataset][:-5]+'_ov.npz'
calculate_statistic_for_ov(save_path)

dataset = 'MERRCoarse'
save_path = config.PATH_TO_LABEL[dataset][:-5]+'_ov.npz'
calculate_statistic_for_ov(save_path)