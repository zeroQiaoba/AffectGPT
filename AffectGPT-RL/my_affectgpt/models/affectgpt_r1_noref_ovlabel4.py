import copy
import einops
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import BertConfig
from my_affectgpt.common.registry import registry
from my_affectgpt.models.blip2 import Blip2Base, disabled_train
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from my_affectgpt.models.Qformer import BertConfig, BertLMHeadModel
from my_affectgpt.models.tokenizer import load_tokenizer_from_LLM
from my_affectgpt.models.encoder import * # only when imported does the registry process take effect
import config

import re
from toolkit.utils.functions import *
from my_affectgpt.evaluation.ew_metric import *
from trl.models import create_reference_model

###########################################################
## delete the ref model and see the effect on results, and whether gpu memory usage can be reduced
## => that is, KL loss is no longer computed
###########################################################
@registry.register_model("affectgpt_r1_noref_ovlabel4")
class AffectGPT_R1_NoRef_OVLabel4(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/affectgpt.yaml",
    }

    def __init__(self, model, cfg):
        super().__init__()

        self.model = model # policy model

        self.tokenizer = self.model.llama_tokenizer

        self.reward_funcs = [self.accuracy_reward, self.format_reward]
        print ('====== use log penalty to pred_num ======')

        self.beta = cfg.beta # balances the weight between KL loss and reward loss
        self.formatbeta = cfg.get("formatbeta", 1.0)
        self.num_generations = cfg.num_generations


    @classmethod
    def from_config(cls, cfg):
        model_cls = registry.get_model_class('affectgpt')
        model = model_cls.from_config(cfg)
        model_cls = cls(model, cfg)
        return model_cls
        
    
    # compute EW-based metrics
    def accuracy_reward(self, target, predicts):
        
        def get_answer(sentence):
            #####################################
            ## remove leading and trailing spaces
            sentence = sentence.strip()
            #####################################
            if sentence.startswith("The character's emotional state is "):
                result = sentence[len("The character's emotional state is "):-1]
                return result
            else:
                return -1

        target_answer = get_answer(target)
        assert target_answer != -1 # the target must satisfy the format requirement

        rewards = []
        for predict in predicts:
            predict_answer = get_answer(predict)
            if predict_answer == -1:
                rewards.append(0)
            else:
                name2gt, name2pred = {}, {}
                name2gt['test']   = string_to_list(target_answer)
                name2pred['test'] = string_to_list(predict_answer)
                level1_fscore, _, _ = wheel_metric_calculation(name2gt=name2gt, name2pred=name2pred, inter_print=False, level='level1')
                final_score = level1_fscore
                ## case1: use final_score directly
                ## case2: restrict the number of pred_labels
                if len(name2gt['test']) >= len(name2pred['test']):
                    reward = final_score
                elif len(name2gt['test']) < len(name2pred['test']):
                    reward = final_score * (np.log(len(name2gt['test'])) / np.log(len(name2pred['test'])))
                rewards.append(reward)
        return rewards


    # format constraint: strictly require the model output to have only one <think></think><answer></answer>
    def format_reward(self, target, predicts):
        """Reward function that checks if the completion has a specific format."""
        pattern = r"<think>.*?</think><answer>.*?</answer>"
        matches = [re.fullmatch(pattern, item, re.DOTALL) for item in predicts]
        return [1.0 if match else 0.0 for match in matches]


    # input_ids      => [q, o_1, o_2]
    # attention_mask => [q, o_1,   0]
    # (input_ids, attention_mask) + model => reading only the second half, this is the probability the model
    # assigns to (q, o_{<t}) -> o_t, which is part of what the formula needs
    # with the input and the output both given, it simply computes the probability of that output
    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, samples):
        
        logits = model.forward_logits(samples)
        logits = logits[:, :-1, :]    # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = samples['input_ids']
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids): # batch -> sample process
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)

        per_token_logps = torch.stack(per_token_logps)

        return per_token_logps



    def forward(self, samples):

        # print (samples['videopaths']) # "samplenew3_00015994"
        ########## samples => 8 outputs (prompt_ids, prompt_masks, completion_ids) ##########
        completion_ids = self.model.forward_sampling(samples, self.num_generations, max_new_tokens=60)    # since there is only ovlabel, generate_ids can be set very small

        ########## filter out invalid generations containing special placeholder tokens ##########
        special_token_strs = [
            config.DEFAULT_FRAME_PATCH_TOKEN,
            config.DEFAULT_FACE_PATCH_TOKEN,
            config.DEFAULT_AUDIO_PATCH_TOKEN,
            config.DEFAULT_MULTI_PATCH_TOKEN,
            config.DEFAULT_IMAGE_PATCH_TOKEN,
        ]
        special_ids = torch.tensor(
            [self.tokenizer.get_vocab()[s] for s in special_token_strs],
            device=completion_ids.device,
        )  # (5,)
        # check whether each completion contains a special token
        has_special = (completion_ids[..., None] == special_ids).any(dim=-1).any(dim=-1)  # (B*G,)
        valid_mask = ~has_special
        # print(f"Valid completions: {valid_mask.sum().item()}/{valid_mask.size(0)} (removed {(~valid_mask).sum().item()})")

        if valid_mask.sum() == 0:
            # when all are invalid, skip this sample and exclude it from the loss
            return {"loss": torch.tensor(0.0, device='cuda', requires_grad=True), "avg_reward": torch.tensor(0.0, device='cuda'), "skip": True}

        completion_ids = completion_ids[valid_mask]
        # print (self.tokenizer.batch_decode(completion_ids, add_special_tokens=False)) # DEBUG
        prompt_ids   = samples['prompt_ids'].repeat_interleave(self.num_generations, dim=0)
        prompt_masks = samples['prompt_masks'].repeat_interleave(self.num_generations, dim=0)
        prompt_ids = prompt_ids[valid_mask]
        prompt_masks = prompt_masks[valid_mask]
        prompt_length = prompt_ids.size(1)


        ########## completion_mask ##########
        stop_token = self.tokenizer('###')['input_ids'][0]
        is_eos = completion_ids == stop_token # [8, 83] find the <|im_end|> after thinking; there is only one such char, not <|endoftext|>
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device='cuda') # initialize eos_idx [83, 83, 83, 83, 83, 83, 83, 83]
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)] # locate the position of the stop token in completion_ids
        sequence_indices = torch.arange(is_eos.size(1), device='cuda').expand(is_eos.size(0), -1) # [8, 83] stores values 0~82
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int() # convert to a 0/1 matrix; the meaningful part mask=1, the meaningless part mask=0


        ########## repeat 8 times ##########
        prompt_completion_masks = torch.cat([prompt_masks, completion_mask], dim=1)  # (B*G_eff, P+C)
        prompt_completion_ids   = torch.cat([prompt_ids,   completion_ids],  dim=1)
        samples['attention_masks'] = prompt_completion_masks
        samples['input_ids']       = prompt_completion_ids
        for keyname in ['faces', 'raw_faces', 'frames', 'raw_frames', 'audios', 'raw_audios', 'images', 'raw_images']:
            if keyname in samples:
                samples[keyname] = torch.cat([samples[keyname]] * self.num_generations, dim=0)
                samples[keyname] = samples[keyname][valid_mask]


        ########## KL loss computation ##########
        # policy model
        per_token_logps = self._get_per_token_logps(self.model, samples)
        per_token_logps = per_token_logps[:, prompt_length - 1 :] # probs corresponding to complete_ids


        ########## reward calculation ##########
        completions = self.tokenizer.batch_decode(completion_ids, add_special_tokens=False)
        completions = [item.split('###')[0] for item in completions]
        rewards_per_func = torch.zeros(len(completions), len(self.reward_funcs), device='cuda')
        for i, reward_func in enumerate(self.reward_funcs):
            output_reward_func = reward_func(target=samples['answers'][0], predicts=completions)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device='cuda')

        rewards = rewards_per_func[:, 0] # optimize using only the ew reward
        avg_reward = torch.mean(rewards)

        # the number of samples after filtering is not fixed, so no more advantage normalization by fixed groups; use reward directly as advantage
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)


        ########## combine into a token-wise loss and backpropagate to optimize **model** ##########
        ## when all advantages are the highest, no optimization; when all are the lowest, no optimization; when advantages are mixed high and low, increase the likelihood of high-reward results and suppress low-reward ones
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) # x - x.detach() allows for preserving gradients from x
        per_token_loss = -per_token_loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() # token-wise loss -> sample-wise loss; size=[1]
        return {"loss": loss, 'avg_reward':avg_reward}
