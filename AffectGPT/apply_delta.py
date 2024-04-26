import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def apply_delta(base_model_path, target_model_path, delta_path):

    print("Loading base model")
    base = AutoModelForCausalLM.from_pretrained(base_model_path, 
    											torch_dtype=torch.float16, 
    											low_cpu_mem_usage=True) # low_cpu_mem_usage=True -> pip install accelerate

    # change to 32001 tokens => add one token and set initial embedding to 0
    DEFAULT_PAD_TOKEN = "[PAD]"
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    num_new_tokens = base_tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN)) # 32000 -> 32001
    base.resize_token_embeddings(len(base_tokenizer))
    input_embeddings = base.get_input_embeddings().weight.data
    output_embeddings = base.get_output_embeddings().weight.data
    input_embeddings[-num_new_tokens:] = 0  # set embedding for new token to zero vector
    output_embeddings[-num_new_tokens:] = 0 # set embedding for new token to zero vector

    print("Loading delta")
    delta = AutoModelForCausalLM.from_pretrained(delta_path, 
    											 torch_dtype=torch.float16, 
    											 low_cpu_mem_usage=True)
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path) # 32001

    print("Applying delta")
    for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    print("Saving target model") # save: automatic split into multiple bins
    base.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


# python stable-vicuna-13b-delta/apply_delta.py --base-model-path='llama-7b-hf'  --delta-path='./vicuna-7b-delta-v0'      --target-model-path='vicuna-7b-v0'     
# python stable-vicuna-13b-delta/apply_delta.py --base-model-path='llama-7b-hf'  --delta-path='./vicuna-7b-delta-v1.1'    --target-model-path='vicuna-7b-v1.1'    
# python stable-vicuna-13b-delta/apply_delta.py --base-model-path='llama-13b-hf' --delta-path='./stable-vicuna-13b-delta' --target-model-path='stable-vicuna-13b' 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)