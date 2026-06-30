import json
import sys
from pathlib import Path
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
per_prompt_rewards_list = []
def main() -> None:
    # start server
    server = VLLMServer(model_id="allenai/OLMo-2-0425-1B",gpu=0)
    server.start()

    # get prompts
    prompt_filenames = ['question_only.prompt', 'r1_zero.prompt', 'r1_zero_three_shot_gsm8k.prompt']
    format_prompt_list = extract_format_prompts(prompt_filenames)
    # ctr=0

    for idx,format_prompt in enumerate(format_prompt_list):
        prompt_filename = prompt_filenames[idx]
        gsm8k_prompts_list,gsm8k_answers = format_gsm8k_prompts(format_prompt) 
        
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

        # smoke test, subset 4
        gsm8k_prompts_list = gsm8k_prompts_list[:4]
        gsm8k_answers = gsm8k_answers[:4]
        

        # get completions from vLLM (in vLLM object form with text, token_ids, finish_reason fields)
        gsm8k_completions = server.generate_completions(gsm8k_prompts_list,sampling_params)
        gsm8k_responses = [completion.text for completion in gsm8k_completions]
        rewards_list = []
        for response,answer in zip(gsm8k_responses,gsm8k_answers,strict=True):
            if "r1" in prompt_filename:
                response_rewards = r1_zero_reward_fn(response,answer)
            else:
                response_rewards = question_only_reward_fn(response,answer) 
            rewards_list.append(response_rewards)

        # score the completions
        print(f"DEBUG PROMPT for {prompt_filename}\n", gsm8k_prompts_list[0])
        print(f"DEBUG LABEL for {prompt_filename}\n", gsm8k_answers[0])
        print(f"DEBUG PREDICTION for {prompt_filename}\n", gsm8k_responses[0])

        print(f'DEBUG REWARDS for {prompt_filename}\n',rewards_list[0])
        per_prompt_rewards_list.append(rewards_list)
        # if ctr==1:
        #     print(gsm8k_prompts_list[0])
        #     print(gsm8k_answers[0])
        #     sys.exit(1)
        # ctr+=1
    server.stop()


    # Download Dataset

    # Set up inference engine 
    # Trial responses

    # evaluate model



if __name__ == "__main__":
    main()