import json
import sys
import torch
import einops
from pathlib import Path
from typing import Any, Callable, Literal
from collections import defaultdict, Counter
from transformers import PreTrainedTokenizerBase
from cs336_alignment.checkpoint import get_model_and_tokenizer
from torch.nn.functional import log_softmax
# from cs336_alignment.vllm_utils import VLLMServer,VLLMCompletion
# from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn 


def tokenize_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer: PreTrainedTokenizerBase
) -> dict[str,torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    
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


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    # Note: two things to remember, index needs to have same number of dims of target tensor, achieved by unsqueeze, squeeze AND torch.gather(target,dim=-1, index=index)
    logits = model.forward(input_ids)["logits"]
    logprobs = log_softmax(logits,dim=-1)
    logprobs_indexed = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze()

    output_dict = {
        "log_probs":logprobs_indexed,
        }

    if return_token_entropy:
        p = logprobs.exp()
        H = -(p*logprobs).sum(dim=-1) # Entropy: elementwise multiply of probs by logprobs for each batch,seq, vocab item, then sum about the vocab dim. X.sum reduces from (b,s,v)->(b,s).
        output_dict["token_entropy"] = H
           
    return output_dict