import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer,VLLMCompletion
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn 


def extract_format_prompts(prompt_filenames):
    
    prompt_list = []
    for prompt_string in prompt_filenames:
        path = Path(__file__).parent / "prompts" / prompt_string
        with open(path, "r") as f:
            prompt_list.append(f.read())    
    return prompt_list

# need to extract the gsm8k answers from the jsonl
def format_gsm8k_prompts(format_prompt, split="test"):
    
    # either get the path for train or test
    gsm8k_path = Path(__file__).parent.parent / "data" / "gsm8k" / f"{split}.jsonl"
    
    # initialize the prompts to pass back and answers for scoring.
    gsm8k_prompts = []
    gsm8k_answers = [] # need to double check form of answer to pass to VLLM server
    
    # load all the examples and format them
    with open(gsm8k_path, "r", encoding='utf-8') as file:
        for line in file:
            line_data = json.loads(line)
            
            # parse question
            raw_question = line_data.get('question','')

            # replace the {question} part of the prompt
            formatted_prompt = format_prompt.replace("{question}", raw_question.strip())

            # add it to the list
            gsm8k_prompts.append(formatted_prompt)

            # parse answers. This post process is specific to gsm8k.
            raw_answer = line_data.get('answer','')
            numerical_answer = raw_answer.split("####")[1].strip() #split by #### and get second element
            gsm8k_answers.append(numerical_answer)

    return(gsm8k_prompts,gsm8k_answers)

# need to extract the gsm8k questions from jsonl and insert into the correct location.
def main() -> None:
    # start server
    server = VLLMServer(model_id="allenai/OLMo-2-0425-1B",gpu=0)
    server.start()

    # get prompts
    prompt_filenames = ['question_only.prompt', 'r1_zero.prompt', 'r1_zero_three_shot_gsm8k.prompt']
    format_prompt_list = extract_format_prompts(prompt_filenames)

    # track responses per prompt by type of reward
    per_prompt_responses = defaultdict(list)
    for idx,format_prompt in enumerate(format_prompt_list):
        
     
        prompt_filename = prompt_filenames[idx]
        gsm8k_prompts_list,gsm8k_answers = format_gsm8k_prompts(format_prompt) 
        prompt_filename_key = prompt_filename.removesuffix(".prompt")
        
        # initialize prompt specific dict, and categorize responses by given categories.
        per_prompt_responses[prompt_filename_key] = {"correct_all":[],"correct_format_only":[],"correct_answer_only":[],"none_correct":[]}

        # initialize sampling params
        sampling_params = {
            "temperature"   :1.0,
            "max_tokens"    :512,
            "n"             :1,
            "seed"          :42
        }
        if "r1" in prompt_filename:
            sampling_params['stop'] = ["</answer>"]
            sampling_params['include_stop_str_in_output'] = True
        # note: top_p is default 1.0 in vLLM so not set.

        # get completions from vLLM (in vLLM object form with text, token_ids, finish_reason fields)
        gsm8k_completions = server.generate_completions(gsm8k_prompts_list,sampling_params)
        gsm8k_responses = [completion.text for completion in gsm8k_completions]
        
        # score the completions
        for response,answer in zip(gsm8k_responses,gsm8k_answers,strict=True):
            if "r1" in prompt_filename:
                response_rewards = r1_zero_reward_fn(response,answer)
            else:
                response_rewards = question_only_reward_fn(response,answer) 
        
            # sort responses reward type
            if response_rewards['format_reward'] == 1 and response_rewards['answer_reward'] == 1:
                per_prompt_responses[prompt_filename_key]["correct_all"].append(response+f"\nLABELED ANSWER: {answer}")
            elif response_rewards['format_reward'] == 1 and response_rewards['answer_reward'] == 0:
                per_prompt_responses[prompt_filename_key]["correct_format_only"].append(response+f"\nLABELED ANSWER: {answer}")
            elif response_rewards['format_reward'] == 0 and response_rewards['answer_reward'] == 1:
                per_prompt_responses[prompt_filename_key]["correct_answer_only"].append(response+f"\nLABELED ANSWER: {answer}")
            else:
                per_prompt_responses[prompt_filename_key]["none_correct"].append(response+f"\nLABELED ANSWER: {answer}")

    server.stop()
    for prompt_name_key,reward_type_dict in per_prompt_responses.items():
        print("\n","prompt_type",prompt_name_key)

        for reward_type, responses in reward_type_dict.items():
            print("\n",f"reward type: {reward_type}")
            print("\n",f"number of examples: {len(responses)}")

            # if reward_type == "correct_format_only":
            #     print("\n Ten Format Correct Only Responses:\n",responses[:10])
            # if reward_type == "none_correct":
            #     print("\n Ten Format None Correct Responses:\n",responses[:10])

if __name__ == "__main__":
    main()


"""
Answers to questions:
(a) How many model generations fall under (1) both corrrect (2) format only reward (3) none corrrect
    For question only: (1) 1 examples (2) 132 examples (3) 1186 examples
    For r1 zero prompt (1) 0 examples (2) 814 examples (3) 505 examples
    For r1 3-shot prpt (1) 215 examples (2) 1057 examples (3) 47 examples

    Observing 10 exampoes from 
        category 2 for question only: its totally garbage output usually.
        category 2 for r1 zero, there are some examples where there is are correct but formatted incorrectly:  "<answer> He would run 3 * 3 = 9 sprints in a week, totaling 9 * 60 = 540 meters. </answer>\nLABELED ANSWER: 540"
        category 2 for r1 3-shot prompt, not many examples are actually correct, its a true issue. 

        for category 3 on r1 prompts there is more occurances of missed positives (actually correct) but the limitation is in parsing when the model messes up the tag "</ think>" instead of "</think>" 
        
(b) How does a model behave depending on the prompt
    question only responses tend to drift and have random text. example: "$$\\boxed{{}}$$,,,,,,,,,You must be logged in to post a comment"
    r1 prompts start a chain of thought that is more reasonable, and the 3-shot further improves upon this. "riginally, the mechanic repaired 6 truck tires for $60 each, so that's 6 * 60 = $360"
"""