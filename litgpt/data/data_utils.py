
import torch

from typing import Any, Callable, Dict, List, Optional, Union
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer

USER_TOKEN = "[USER]"
SYSTEM_TOKEN = "[SYSTEM]"
KNOW_TOKEN = "[KNOW]"
PATH_TOKEN = "[PATH]"
SEP_TOKEN = "[SEP]"
PROFILE_TOKEN = "[PROFILE]"
CONTEXT_TOKEN = "[CONTEXT]"
GOAL_TOKEN = "[GOAL]"
TARGET = "[TARGET]"
TOPIC_TOKEN = "[TOPIC]"

# for user side information
USER_GOAL_TOKEN = "[USER_GOAL]"
USER_TOPIC_TOKEN = "[USER_TOPIC]"
USER_SITUATION_TOKEN = "[SITUATION]"
USER_EMOTION_TOKEN = "[EMOTION]"
USER_ACTION_TOKEN = "[ACTION]"
USER_KNOWLEDGE_TOKEN = "[KNOWLEDGE]"


def unique_ordered_list(lst: list):
    all_items = list(set(lst))
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    assert len(result) == len(all_items), "The list is illenegal"
    return result

def convert_example_to_feature_for_convint(instance: dict, 
                                           prompt_style: Union[str, PromptStyle],
                                           tokenizer: Tokenizer,
                                           max_seq_length: int,
                                           mask_prompt: bool,
                                           ignore_index: int,
                                           is_test: bool = False):
    """
    function that convert an instance to input and labels for a response generation model.
    @param instance: an instance from the data.
    @return: an input sequence and its corresponding labels.
    """

    dialogue_context = instance['dialogue_context']
    dialogue_str = ""
    prev_convints = instance['pre_gpt-3.5-intention'][1:]

    idx = 0
    num_prev_convints = len(prev_convints)

    for utt in dialogue_context:
        
        if utt['role'] == "user":
            dialogue_str += USER_TOKEN + " "
            dialogue_str += utt['content'] + " "

            if idx < num_prev_convints:
                dialogue_str += USER_SITUATION_TOKEN + " " + prev_convints[idx][USER_SITUATION_TOKEN] + " "
                dialogue_str += USER_EMOTION_TOKEN + " " + prev_convints[idx][USER_EMOTION_TOKEN] + " "
                dialogue_str += USER_ACTION_TOKEN + " " + prev_convints[idx][USER_ACTION_TOKEN] + " "
                dialogue_str += USER_KNOWLEDGE_TOKEN + " " + prev_convints[idx][USER_KNOWLEDGE_TOKEN] + " "
                idx += 1
        
        elif utt['role'] == 'assistant':
            dialogue_str += SYSTEM_TOKEN + " "
            dialogue_str += utt['content'] + " "

    target = instance['task_background']['target_topic']

    input_str = f"{CONTEXT_TOKEN}: {dialogue_str}"

    # construct the label for response generation task
    if not is_test:
        label_str = USER_SITUATION_TOKEN + " " + instance['gpt-3.5-intention'][USER_SITUATION_TOKEN] + " "
        label_str += USER_EMOTION_TOKEN + " " + instance['gpt-3.5-intention'][USER_EMOTION_TOKEN] + " "
        label_str += USER_ACTION_TOKEN + " " + instance['gpt-3.5-intention'][USER_ACTION_TOKEN] + " "
        label_str += USER_KNOWLEDGE_TOKEN + " " + instance['gpt-3.5-intention'][USER_KNOWLEDGE_TOKEN] 
    else:
        label_str = USER_SITUATION_TOKEN + " " + instance['gpt-4-intention'][USER_SITUATION_TOKEN] + " "
        label_str += USER_EMOTION_TOKEN + " " + instance['gpt-4-intention'][USER_EMOTION_TOKEN] + " "
        label_str += USER_ACTION_TOKEN + " " + instance['gpt-4-intention'][USER_ACTION_TOKEN] + " "
        label_str += USER_KNOWLEDGE_TOKEN + " " + instance['gpt-4-intention'][USER_KNOWLEDGE_TOKEN] 
    
    prompt = prompt_style.apply(prompt=input_str)
    
    encoded_prompt = tokenizer.encode(prompt, max_length=max_seq_length)
    encoded_response = tokenizer.encode(label_str, bos=False, eos=True, max_length=max_seq_length)
    encoded_prompt_and_response = torch.cat((encoded_prompt, encoded_response)).type(torch.int64)
    if max_seq_length > 0:
        encoded_prompt_and_response = encoded_prompt_and_response[: max_seq_length]

    labels = encoded_prompt_and_response.clone()
    if mask_prompt:
        labels[: len(encoded_prompt)] = ignore_index
    
    # if len(dialogue_context) <= 5:
        
    #     print("=====================================================")
    #     print("input_str: ")
    #     print(input_str)

    #     print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #     print("label_str: ")
    #     print(label_str)

    #     print("decoded_label_str: ")
    #     print(tokenizer.decode(encoded_response))

    #     print("encoded_prompt_and_response: ")

    #     print(encoded_prompt_and_response)

    #     print("labels: ")
    #     print(labels)

    #     print("decoded_masked_prompt_labels: ")
    #     print(tokenizer.decode(torch.where(labels == ignore_index, torch.tensor(tokenizer.bos_id), labels)))

    #     print('test')
    #     print(tokenizer.decode(torch.tensor([518]))) # [

    #     print("-----------------------------------------------------")
    #     print("prompt: ")
    #     print(prompt)

    #     print("*****************************************************")
    #     print(f"max_seq_length: {max_seq_length}, ignore_index: {ignore_index}")

    #     print("#####################################################")

    #     print("decode_prompt: ")
    #     print(tokenizer.decode(encoded_prompt))

    #     assert tokenizer.decode(encoded_prompt) == prompt, "The decoded prompt is not correct."

        

        
    # else:
    #     exit()

    features = {
        "dialogue_context": input_str,
        "label_str": label_str,
        "input_ids": encoded_prompt_and_response,
        "input_ids_wo_response": encoded_prompt,
        "labels": labels
    }

    return features