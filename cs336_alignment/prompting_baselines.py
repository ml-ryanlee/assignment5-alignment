import json
import sys
from pathlib import Path
from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer,VLLMCompletion
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn 


def extract_format_prompts():
    prompt_filenames = ['question_only.prompt', 'r1_zero.prompt', 'r1_zero_three_shot_gsm8k.prompt']
    prompt_list = []
    for prompt_string in prompt_filenames:
        path = Path(__file__).parent / "prompts" / prompt_string
        with open(path, "r") as f:
            prompt_list.append(f.read())    
    return prompt_list

# need to extract the gsm8k answers from the jsonl
def format_gsm8k_prompts(format_prompt,r1=False,split="test"):
    
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
            if r1:
                # format for r1 prompt, by replacing the user {question} with the actual prompt
                formatted_prompt = format_prompt.replace("{question}", raw_question.strip())
            else:
                formatted_prompt = format_prompt+raw_question.strip()
            gsm8k_prompts.append(formatted_prompt)

            # parse answers
            raw_answer = line_data.get('answer','')
            numerical_answer = raw_answer.split("####")[1].strip() #split by #### and get second element
            gsm8k_answers.append(numerical_answer)

    return(gsm8k_prompts,gsm8k_answers)


    


# need to extract the gsm8k questions from jsonl and insert into the correct location.



def main() -> None:
    # start server
    # server = VLLMServer(model_id="allenai/OLMo-2-0425-1B")
    # server.start

    # get prompts
    format_prompt_list = extract_format_prompts()
    
    for format_prompt in format_prompt_list:

        gsm8k_prompts_list = format_gsm8k_prompts(format_prompt) 

        print(gsm8k_prompts_list[0],"\n")

    # Download Dataset

    # Set up inference engine 
    # Trial responses

    # evaluate model



if __name__ == "__main__":
    main()