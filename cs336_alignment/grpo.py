import json
import sys
import torch
import einops
from pathlib import Path
from typing import Any, Callable, Literal
from collections import defaultdict, Counter
from einops import rearrange
# from transformers import PreTrainedTokenizerBase
# from cs336_alignment.checkpoint import get_model_and_tokenizer
from torch.nn.functional import log_softmax

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    rollout_batch_size = len(rollout_responses)
    reward_list = []
    reward_sum = 0
    format_reward_sum = 0
    for i in range(rollout_batch_size):
        rewards_dict = reward_fn(rollout_responses[i],repeated_ground_truths[i])
        reward = rewards_dict["reward"]
        reward_list.append(reward)
        reward_sum += reward
        
        format_reward = rewards_dict["format_reward"]
        format_reward_sum += format_reward

    metadata = {
        "average_total_reward": (reward_sum/rollout_batch_size),
        "average_format_reward": (format_reward_sum/rollout_batch_size)
    }

    reward_tensor = torch.tensor(reward_list)

    return(reward_tensor, metadata)

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """

    # rearrange raw_rewards from unrolled to 2D
    raw_rewards = rearrange(raw_rewards, '(n_prompts_per_rollout_batch group_size) -> n_prompts_per_rollout_batch group_size', group_size=group_size)
    # print("Raw rewards shape:", raw_rewards.shape)
    
    mean_reward = raw_rewards.mean(dim=-1,keepdim=True) 
    # print("Mean Reward shape:", mean_reward.shape)
    
    std_reward = raw_rewards.std(dim=-1,keepdim=True)  
    # print("std reward shape: ", std_reward.shape)

    metadata = {
        "group_average_reward": mean_reward,
        "group_std_reward": std_reward,
        "group_size": group_size
    }


    # advantage calculation
    if baseline == "mean":
        advantages = raw_rewards - mean_reward
    else:
        raise NotImplementedError

    # advantage normalization    
    if advantage_normalizer == "std":
        advantages = advantages / (std_reward+advantage_eps)
    else:
        raise NotImplementedError
    
    # reshape into unrolled 1D vector
    advantages = rearrange(advantages, 'n_prompts_per_rollout_batch group_size -> (n_prompts_per_rollout_batch group_size)', group_size=group_size)
    return (advantages,metadata)

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    # 
    if importance_reweighting_method != "none":
        raise NotImplementedError

    per_token_policy_gradient_loss = -policy_log_probs * raw_rewards_or_advantages # [b,seq] * [b,1] 
    metadata = {
        "clip_fraction":None # placeholder
    }
    return (per_token_policy_gradient_loss, metadata)

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    loss = per_token_policy_gradient_loss

    # apply the mask
    loss = loss * mask

    # step 1, count the ones in the mask for seq normalization
    if loss_normalization == "sequence":
        seq_lengths = torch.sum(mask,dim=-1,keepdim=True)
        # average seq with unmasked lengths
        loss = loss / seq_lengths

    elif loss_normalization == "constant":
        assert normalization_constant is not None, "Error: if using constant normalization, normalization_constant cannot be None"
        # average seq with a constant
        loss = loss / normalization_constant

    else: 
        raise NotImplementedError(f"Unknown loss_normalization: {loss_normalization}")

    # Average across batch
    loss = torch.sum(loss,dim=-1) # sum over rows (since token scalar values were individually averaged)
    loss = torch.mean(loss) # get average across rows
    return loss