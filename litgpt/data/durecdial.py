# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import re
import json
import copy
from loguru import logger
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import DataLoader, random_split

from litgpt.data import DataModule, ConvINTSFTDataset, get_sft_collate_fn
from litgpt.data import unique_ordered_list
from litgpt.data import convert_example_to_feature_for_convint
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer


@dataclass
class DuRecDial(DataModule):
    """Alpaca data module for supervised finetuning."""

    mask_prompt: bool = True
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    prompt_style: Union[str, PromptStyle] = "durecdial"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[ConvINTSFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[ConvINTSFTDataset] = field(default=None, init=False, repr=False)

    target_goals: list = field(default_factory=lambda: [
        "Movie recommendation",
        "Food recommendation",
        "Music recommendation",
        "POI recommendation",
    ])

    def __post_init__(self) -> None:
        super().__init__()
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(self, 
                train_data_path: Path,
                dev_data_path: Path,
                test_data_path: Path,
                train_mid_convint_path: Path,
                dev_mid_convint_path: Path,
                test_mid_convint_path: Path,
                dev_high_convint_path: Path,
                test_high_convint_path: Path,
                tokenizer: Optional[Tokenizer] = None, 
                batch_size: int = 1, 
                max_seq_length: Optional[int] = None) -> None:
        
        self.train_data_path = train_data_path
        self.dev_data_path = dev_data_path
        self.test_data_path = test_data_path

        self.train_mid_convint_path = train_mid_convint_path
        self.dev_mid_convint_path = dev_mid_convint_path
        self.test_mid_convint_path = test_mid_convint_path

        self.dev_high_convint_path = dev_high_convint_path
        self.test_high_convint_path = test_high_convint_path

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:

        self.topics = []
        self.goals = []
        
        assert self.train_data_path.exists(), f"Train data file not found at {self.train_data_path}"
        self.train_convs = self.pipline(self.train_data_path)
        self.train_convs = self.load_llm_convint(self.train_convs, self.train_mid_convint_path, "gpt-3.5-turbo")

        assert self.dev_data_path.exists(), f"Dev data file not found at {self.dev_data_path}"
        self.dev_convs = self.pipline(self.dev_data_path)
        self.dev_convs = self.load_llm_convint(self.dev_convs, self.dev_mid_convint_path, "gpt-3.5-turbo")
        self.dev_convs = self.load_llm_convint(self.dev_convs, self.dev_high_convint_path, "gpt-4-turbo-2024-04-09")

        assert self.test_data_path.exists(), f"Test data file not found at {self.test_data_path}"
        self.test_convs = self.pipline(self.test_data_path)
        self.test_convs = self.load_llm_convint(self.test_convs, self.test_mid_convint_path, "gpt-3.5-turbo")
        self.test_convs = self.load_llm_convint(self.test_convs, self.test_high_convint_path, "gpt-4-turbo-2024-04-09")

        self.train_instances = []
        self.dev_instances = []
        self.test_instances = []

    def pipline(self, data_path: Path):
        # read data
        with open(data_path, 'r') as f:
            convs = f.readlines()
            assert len(convs) > 0
        # repurpose data
        convs = self.repurpose_dataset(convs)
        return convs
    
    def repurpose_dataset(self, data):
        """convert the original goal-driven setting to the target-driven CRS setting.
        only consider recommendation-oriented conversations including food, movie, music, poi recommendation

        Args:
            data (_type_): list of json strings, each element is a conversation.

        Returns:
            _type_: list of dictionary each element corresponds to a repurposed conversation,
        """
        new_data = []
        new_conv_id = 0
        for line in data:
            line = json.loads(line)
            # each line is a conversation
            # in each conversation we have user_profile, goal_sequencs, topics_sequences and conversations.
            scenario = line['goal']
            steps = scenario.split('-->')

            # get the target goal and target topic
            i = len(steps) - 1
            while i >= 0 and ("Say goodbye" in steps[i] or 'recommendation' not in steps[i]):
                i = i - 1

            # we can not find the target recommendation goal
            if i < 0:
                continue

            # preprocessing to get the target goal and the target topic
            target_goal = re.sub(r'\(.*?\)', '', steps[i]).replace(')', '').strip()
            target_topic = steps[i].replace(target_goal, "")[1:-1].strip()

            # there are some cases such as "A. B", B is the accepted item therefore we want to get B.
            if len(target_topic.split('、')) == 2:
                target_topic = target_topic.split('、')[-1].strip()

            target_goal = re.sub(r'[0-9]', '', target_goal).replace("[]", '').strip()
            # if the target goal is not in our considered target list.
            assert target_goal in self.target_goals
            line['conv_id'] = new_conv_id
            line['target_goal'] = target_goal
            line['target_topic'] = target_topic
            new_data.append(line)
            new_conv_id += 1
        return new_data
    
    def load_llm_convint(self, convs: list, llm_convint_path: Path, llm_model_type: str) -> list:
        """method that loads the llm convints from the file.
        """
        assert llm_convint_path.exists(), f"LLM convint file not found at {llm_convint_path}"
        with open(llm_convint_path, 'rb') as f:
            convint_data = f.readlines()

        all_llm_convints = []
        for line in convint_data:
            line = json.loads(line)
            all_llm_convints.append(line)

        for conv in convs:
            conv_id = conv['conv_id']

            for llm_convints in all_llm_convints:
                if llm_convints['conv_id'] == conv_id:
                    conv[llm_model_type] = llm_convints[llm_model_type]
                    break
        logger.info(f"Loaded {llm_model_type} convints from {llm_convint_path}.")
        return convs

    
    def process_convs(self):

        self.train_instances = self.construct_instances(self.train_convs)
        self.dev_instances = self.construct_instances(self.dev_convs)
        self.test_instances = self.construct_instances(self.test_convs)

        goal_dict = defaultdict(int)
        # log goal count
        for goal in self.goals:
            goal_dict[goal] += 1

        # log target item w.r.t data split
        train_target_items = []
        dev_target_items = []
        test_target_items = []

        # log target w.r.t different domains
        movie_target_items = defaultdict(list)
        music_target_items = defaultdict(list)
        food_target_items = defaultdict(list)
        poi_target_items = defaultdict(list)

        for instance in self.train_instances:
            train_target_items.append(instance['task_background']['target_topic'])

            if instance['task_background']['target_goal'] == 'Movie recommendation':
                movie_target_items['train'].append(instance['task_background']['target_topic'])
            if instance['task_background']['target_goal'] == 'Music recommendation':
                music_target_items['train'].append(instance['task_background']['target_topic'] )
            if instance['task_background']['target_goal'] == 'Food recommendation':
                food_target_items['train'].append(instance['task_background']['target_topic'] )

            if instance['task_background']['target_goal'] == 'POI recommendation':
                poi_target_items['train'].append(instance['task_background']['target_topic'] )

        for instance in self.dev_instances:
            dev_target_items.append(instance['task_background']['target_topic'])

            if instance['task_background']['target_goal'] == 'Movie recommendation':
                movie_target_items['dev'].append(instance['task_background']['target_topic'] )
            if instance['task_background']['target_goal'] == 'Music recommendation':
                music_target_items['dev'].append(instance['task_background']['target_topic'] )
            if instance['task_background']['target_goal'] == 'Food recommendation':
                food_target_items['dev'].append(instance['task_background']['target_topic'] )

            if instance['task_background']['target_goal'] == 'POI recommendation':
                poi_target_items['dev'].append(instance['task_background']['target_topic'] )

        for instance in self.test_instances:
            test_target_items.append(instance['task_background']['target_topic'])

            if instance['task_background']['target_goal'] == 'Movie recommendation':
                movie_target_items['test'].append(instance['task_background']['target_topic'] )
            if instance['task_background']['target_goal'] == 'Music recommendation':
                music_target_items['test'].append(instance['task_background']['target_topic'] )
            if instance['task_background']['target_goal'] == 'Food recommendation':
                food_target_items['test'].append(instance['task_background']['target_topic'] )

            if instance['task_background']['target_goal'] == 'POI recommendation':
                poi_target_items['test'].append(instance['task_background']['target_topic'] )

        logger.info(
            f"Statistics by data splits: Train: {len(list(set(train_target_items)))}, Dev: {len(list(set(dev_target_items)))}, Test: {len(list(set(test_target_items)))}")

        for t in ['train', 'dev', 'test']:
            logger.info(
                f"Statistics by domain splits {t}: Movie: {len(list(set(movie_target_items[t])))}, Music: {len(list(set(music_target_items[t])))}, Food: {len(list(set(food_target_items[t])))}, POI: {len(list(set(poi_target_items[t])))}")

        # for fixing the order of goals and topics
        self.goals = unique_ordered_list(self.goals)
        self.topics = unique_ordered_list(self.topics)

        infor_dict = {
            "Num_topics": len(self.topics),
            "Num_goals": len(self.goals),
            "Train_convs": len(self.train_convs),
            "Dev_convs": len(self.dev_convs),
            "Test_convs": len(self.test_convs),
            "Train_instances": len(self.train_instances),
            "Dev_instances": len(self.dev_instances),
            "Test_instances": len(self.test_instances)

        }

        logger.info("### ==================== Dataset Details ==================== ###")
        logger.info(f"    Dataset = Durecdial")
        for k, v in infor_dict.items():
            logger.info(f"    {k} = {v}")
    
    
    def construct_instances(self, data):
        """method that process the conversations to get input instances.
        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        all_instances = []
        for line in data:
            conv_id = line['conv_id']
            instances = self.construct_instances_from_conv(conv_id, line)
            all_instances.extend(instances)
        return all_instances
    

    def construct_instances_from_conv(self, conv_id, conv):
        """ method that constructs input examples from a conversation
        each instance consists of task background, dialogue context and its corresponding response.

        Args:
            conv_id (_type_): the index of the input conversation
            conv (_type_): the conversation

        Returns:
            _type_: list of input instances.
        """
        instances = []
        task_background = {
            "target_goal": conv['target_goal'],
            "target_topic": conv['target_topic'],
            "user_profile": conv['user_profile'],
        }
        utts = []
        goals = ["None"]
        topics = ["None"]
        user_goals = ['None']
        user_topics = ['None']
        gpt_3_5_intentions = ["None"]
        gpt_4_intentions = ["None"]
        
        # even for user, and odd for agent.
        role = 0
        if conv['goal_type_list'][0] == "Greetings":
            # agent starts the conversation.
            role = -1
        
        #create the goal, topic paths
        goal_path = []
        topic_path = []

        # add for checking the lengthes of conversations, goals, topics, and knowledge.
        assert len(conv['conversation']) == len(conv['goal_type_list']) == len(conv['goal_topic_list']) == len(conv['knowledge'])
        for idx, (utt, goal, topic, knowledge) in enumerate(list(
                zip(conv['conversation'], conv['goal_type_list'], conv['goal_topic_list'], conv['knowledge']))):
            # user responses.
            self.goals.append(goal)
            self.topics.append(topic)
            
            if role % 2 == 0:
                # add for tracking the user goals and topics.
                gpt_3_5_intention = conv['gpt-3.5-turbo'][role // 2] if 'gpt-3.5-turbo' in conv else "None"
                gpt_4_intention = conv['gpt-4-turbo-2024-04-09'][role // 2] if 'gpt-4-turbo-2024-04-09' in conv else "None"
                    
                
                utts.append({'role': 'user', 'content': utt, 'user_goal': goal, 'user_topic': topic, 
                             'gpt-3.5-intention': gpt_3_5_intention, 'gpt-4-intention': gpt_4_intention})
                user_goals.append(goal)
                user_topics.append(topic)
                gpt_3_5_intentions.append(gpt_3_5_intention)
                gpt_4_intentions.append(gpt_4_intention)
            # the agent starts the conversaiton.
            elif role == -1:
                # add for tracking the asst goals and topics.
                utts.append({'role': 'assistant', 'content': utt, 'asst_goal': goal, 'asst_topic': topic})
                goals.append(goal)
                topics.append(topic)
            # system response
            else:
                # construct the goal, topic path to the target goal, topic
                to_target_goal_path = []
                to_target_topic_path = []
                # tmp_goal = goal
                # tmp_topic = topic
                tmp_idx = idx

                # get the target goal, topic idx:
                tmp = len(conv['goal_topic_list']) - 1
                while tmp > 0 and conv['goal_type_list'][tmp] != task_background['target_goal'] and \
                        conv['goal_topic_list'][tmp] != task_background['target_topic']:
                    tmp -= 1

                # loop until we meet the target goal, topic
                # increase the tmp idx
                # append the goal, topic to the paths.
                while (tmp_idx < tmp):
                    to_target_goal_path.append(conv['goal_type_list'][tmp_idx])
                    to_target_topic_path.append(conv['goal_topic_list'][tmp_idx])
                    # tmp_goal = conv['goal_type_list'][tmp_idx]
                    # tmp_topic = conv['goal_topic_list'][tmp_idx]
                    # shift 2 due to user responses.
                    tmp_idx += 2

                # append the target goal, topic to the end of the list
                # to_target_topic_path.append(task_background['target_topic'])
                # to_target_goal_path.append(task_background['target_goal'])

                if len(to_target_goal_path) == 0 and len(to_target_topic_path) == 0:
                    to_target_goal_path = [goal]
                    to_target_topic_path = [topic]
                

                goal_path = copy.deepcopy(to_target_goal_path)
                topic_path = copy.deepcopy(to_target_topic_path)

                # to_target_goal_path.append(task_background['target_goal'])
                # to_target_topic_path.append(task_background['target_topic'])

                # reverse the lists.
                to_target_goal_path.reverse()
                to_target_topic_path.reverse()

                # constructing an instance.
                instance = {
                    "conv_id": conv_id,
                    "response": utt,
                    "goal": goal,
                    "topic": topic,
                    "user_goal": user_goals[-1],
                    "user_topic": user_topics[-1],
                    "gpt-3.5-intention": gpt_3_5_intentions[-1],
                    "gpt-4-intention": gpt_4_intentions[-1],
                    "reversed_goals": to_target_goal_path,
                    "reversed_topics": to_target_topic_path,
                    "knowledge": knowledge,
                    "pre_goals": copy.deepcopy(goals),
                    "pre_topics": copy.deepcopy(topics),
                    "pre_user_goals": copy.deepcopy(user_goals[:-1]),
                    "pre_user_topics": copy.deepcopy(user_topics[:-1]),
                    "pre_gpt-3.5-intention": copy.deepcopy(gpt_3_5_intentions[:-1]),
                    "pre_gpt-4-intention": copy.deepcopy(gpt_4_intentions[:-1]),
                    "dialogue_context": copy.deepcopy(utts),
                    "task_background": copy.deepcopy(task_background),
                    "goal_path": goal_path,
                    "topic_path": topic_path
                }
                instances.append(instance)
                # add for tracking the asst goals and topics.
                utts.append({'role': 'assistant', 'content': utt, 'asst_goal': goal, 'asst_topic': topic})
                goals.append(goal)
                topics.append(topic)
            role = role + 1
        
        # # assign goal, topic paths to instances
        # for instance in instances:
        #     instance['goal_path'] = goals
        #     instance['topic_path'] = topics
        return instances
    
    
    def setup(self, stage: str = "") -> None:
        
        # to obtain all the instances
        self.process_convs()

        self.train_dataset = ConvINTSFTDataset(
            data=self.train_instances,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
            transform=convert_example_to_feature_for_convint,
            is_test=False,
        )
        self.dev_dataset = ConvINTSFTDataset(
            data=self.dev_instances,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
            transform=convert_example_to_feature_for_convint,
            is_test=True,
        )
        self.test_dataset = ConvINTSFTDataset(
            data=self.test_instances,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
            transform=convert_example_to_feature_for_convint,
            is_test=True,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )



