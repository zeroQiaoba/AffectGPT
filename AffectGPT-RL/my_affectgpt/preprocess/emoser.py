import tqdm
import config
from scipy.io import wavfile
import numpy as np
from toolkit.utils.read_files import *
from toolkit.utils.functions import *
from my_affectgpt.evaluation.ew_metric import *

from toolkit.utils.read_files import *
from my_affectgpt.evaluation.ew_metric import *
from my_affectgpt.evaluation.wheel import *
from my_affectgpt.datasets.builders.image_text_pair_builder import * # load all dataset cls

'''
s05 (3)
'''


#################################################
## Report Sample Numbers for each Dataset
#################################################
def get_dataset_cls(dataset):
    if dataset == 'MER2023':     return MER2023_Dataset()
    if dataset == 'MER2024':     return MER2024_Dataset()
    if dataset == 'MELD':        return MELD_Dataset()
    if dataset == 'IEMOCAPFour': return IEMOCAPFour_Dataset()
    if dataset == 'CMUMOSI':     return CMUMOSI_Dataset()
    if dataset == 'CMUMOSEI':    return CMUMOSEI_Dataset()
    if dataset == 'SIMS':        return SIMS_Dataset()
    if dataset == 'SIMSv2':      return SIMSv2_Dataset()
    if dataset == 'OVMERDPlus':  return OVMERDPlus_Dataset()
    print ('dataset cls not provided!')
   
for dataset in ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFour', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2', 'OVMERDPlus']:
    dataset_cls = get_dataset_cls(dataset)
    test_names = dataset_cls.read_test_names()
    print (f'{dataset} => {len(test_names)}')
