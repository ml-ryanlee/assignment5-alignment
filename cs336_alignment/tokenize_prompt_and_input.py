import json
import sys
import torch
import einops
from pathlib import Path
from collections import defaultdict, Counter
from transformers import PreTrainedTokenizerBase
from cs336_alignment.checkpoint import get_model_and_tokenizer
# from cs336_alignment.vllm_utils import VLLMServer,VLLMCompletion
# from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn 


def tokenize_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer: PreTrainedTokenizerBase
) -> dict[str,torch.Tensor]:
    
    # need input_ids, labels, response_mask
    assert len(prompt_strs) == len(output_strs), "ERROR: prompt and output batch sizes must be identical"

    # track the output start 
    comb_ids_list = []
    comb_ids_idx_list = []
    max_comb_len = 0

    batch_size = len(prompt_strs)
    
    for idx in range(batch_size):
        prompt = prompt_strs[idx]
        prompt_ids = tokenizer.encode(prompt,add_special_tokens=False)
        response_start_idx = len(prompt_ids)

        output = output_strs[idx]
        output_ids = tokenizer.encode(output,add_special_tokens=False)

        comb_ids = prompt_ids + output_ids
        response_end_idx = len(comb_ids) # non-inclusive
        max_comb_len = max(max_comb_len, len(comb_ids))

        # add the combined ids to the list, as well as start and end idx
        comb_ids_list.append(comb_ids)
        comb_ids_idx_list.append((response_start_idx, response_end_idx)) # start of response and end of response idx tracked as tuple

    
    # create padded torch tensors (input_ids, labels, response_mask). We use torch.stack to combine them later
    input_ids_list =[]
    response_mask_list = []
    pad_id = tokenizer.pad_token_id

    # populate the empty torch tensors, row by row
    for idx in range(batch_size):
        comb_ids = comb_ids_list[idx]
        resp_start, resp_end = comb_ids_idx_list[idx] # resp_end not inclusive
        
        # prepare input_ids_tensor with padding
        input_ids = comb_ids[:resp_end]
        num_pad_tokens = max_comb_len-len(input_ids)
        input_ids_padded = input_ids + [pad_id] * num_pad_tokens
        input_ids_tensor = torch.tensor(input_ids_padded) # slice off last token
        input_ids_list.append(input_ids_tensor)

        # prepare response_mask (tricky, align with labels_ids)
        response_mask_ids = [0] * resp_start 
        num_response_tokens = resp_end-resp_start # we are keeping all the response tokens, so the length is unchanged
        response_mask_ids = response_mask_ids + [1] * num_response_tokens
        num_pad_tokens = max_comb_len - len(response_mask_ids)
        response_mask_ids_padded = response_mask_ids + [0] * num_pad_tokens # according to specs, 0 where not response.
        response_mask_ids_tensor = torch.tensor(response_mask_ids_padded)
        response_mask_list.append(response_mask_ids_tensor)

        assert(len(input_ids_tensor)==len(response_mask_ids_tensor)), "ERROR: lengths of input_ids and response_mask tensor should be identical after padding."

    stacked_ids = torch.stack(input_ids_list)
    input_ids= stacked_ids[:,:-1] # slice off the last entry
    labels = stacked_ids[:,1:] # slice off the first entry
    response_mask = torch.stack(response_mask_list)
    response_mask = response_mask[:,1:]

    output_dict = {"input_ids":input_ids,
                   "labels":labels,
                   "response_mask":response_mask}

    return output_dict


    
    


