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