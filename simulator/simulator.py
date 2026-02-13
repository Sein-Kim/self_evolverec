import torch
import numpy as np
import json
from collections import defaultdict
from openai import OpenAI
import os
from datetime import datetime
import simulator.prompts_list as prompts_list


def get_userfeedback(personality, rec_page, interaction_his, results_1):
    return f"""
You excel at role-playing. Picture yourself as a single user exploring a recommendation page like Amazon.
Your goal is to provide realistic behavior and explain the causal reasons behind your choices to help improve the recommendation model.

The goal of this task is NOT UI feedback.
It is to help improve the recommendation model itself by explaining WHY certain items felt right or wrong.

## Your fixed traits
{personality}

## Your recent context (recent interactions)
- {interaction_his}

## Current recommendation list
{rec_page}

------------------------------------------------------------
PART 1 — BEHAVIOR (already completed)
------------------------------------------------------------
You have already evaluated each item and indicated whether you would realistically choose it now:
{results_1}

------------------------------------------------------------
PART 2 — MODEL_FEEDBACK_JSON (model-improvement focused)
------------------------------------------------------------
Analyze the recommendation list from the perspective of *decision reasons*.

Focus on:
- Why the items you liked felt like strong matches.
- Why the items you disliked felt off, even if they looked similar on the surface.
- What signals seem to be overused or underused by the system.
- Where the system generalized correctly vs incorrectly from your recent behavior.

IMPORTANT RULES:
- Do NOT talk about UI, layout, or presentation.
- Do NOT suggest business rules or exposure tricks.
- Do NOT mention model internals or technical terms.
- Speak as a user, but explain your choices clearly and causally.
- Each reason should help explain what kind of similarity or pattern actually matters to you right now.

Output ONLY the following JSON:

MODEL_FEEDBACK_JSON:
{{
  "positive_alignment_reasons": [
    "Concrete reason why several recommended items felt like natural next choices",
    "Another clear reason why the system understood part of your intent"
  ],
  "negative_misalignment_reasons": [
    "Concrete reason why some items missed your intent despite appearing related",
    "Another reason explaining why certain recommendations felt off"
  ],
  "overgeneralized_signals": [
    "A pattern the system seemed to rely on too heavily when recommending items",
    "Another overused pattern"
  ],
  "missing_or_underweighted_signals": [
    "A type of information or nuance from your recent behavior that the system seemed to miss",
    "Another missing nuance"
  ],
  "summary_diagnosis": "One short sentence summarizing how the system interpreted your intent, correctly or incorrectly."
}}
"""

aggregator_prompt = """
You are a Lead Recommender Systems Analyst synthesizing user simulation feedback
to diagnose how the recommendation model is behaving.

Input:
- A list of User Feedback Reports (Json objects) from multiple simulated users:
{per_user_json_list}

Each USER_DIAGNOSIS_JSON reflects:
- what users expected based on recent behavior,
- why certain recommendations felt right or wrong,
- which patterns consistently helped or hurt their willingness to choose items near the top.

Goal:
Produce a concise, system-level diagnosis of:
- what decision patterns the model appears to rely on,
- where those patterns align or misalign with actual user choice logic,
- which misalignments most directly reduce the quality of top-ranked items.

Do NOT discuss UI, layout, exposure strategy, or business rules.
Do NOT propose specific model architectures or losses.
Stay at the level of behavioral decision logic inferred from user feedback.

------------------------------------------------------------
OUTPUT FORMAT (JSON ONLY — all keys required)
------------------------------------------------------------

{{
  "status": "<CRITICAL | NEEDS_IMPROVEMENT | STABLE>",

  "SYSTEM_INTERPRETATION_OF_USER_INTENT": 
    "One short sentence describing how the system seems to interpret user intent overall, based on aggregated feedback.",

  "WHERE_THE_SYSTEM_GETS_IT_RIGHT": [
    "A recurring pattern where the system's recommendations align well with why users would actually choose items.",
    "Another clear alignment pattern (optional)."
  ],

  "WHERE_THE_SYSTEM_GETS_IT_WRONG": [
    "A recurring misinterpretation that causes strong matches to be ranked lower than weaker ones.",
    "Another concrete misalignment between user choice logic and system behavior."
  ],

  "OVERUSED_DECISION_PATTERNS": [
    "A type of similarity or pattern the system appears to rely on too heavily, leading to false positives.",
    "Another overgeneralized pattern (optional)."
  ],

  "UNDERUSED_OR_MISSED_DECISION_PATTERNS": [
    "A type of signal or nuance users repeatedly mention as important but insufficiently reflected in rankings.",
    "Another missing or weakly captured decision factor (optional)."
  ],

  "PRIMARY_FAILURE_MODE_AT_TOP_OF_LIST":
    "One sentence explaining the dominant reason why the top few items often fail to contain enough things users would choose right now.",

  "DIAGNOSTIC_SIGNALS_TO_TRACK": [
    "A measurable signal that would reveal whether the system is ranking genuinely choice-worthy items near the top.",
    "Another signal that would expose overgeneralization or missed intent."
  ]
}}

Rules:
- No numeric scores or internal error codes.
- No UI or exposure language.
- No raw user quotes; always generalize and compress.
- Every statement should describe a repeatable system behavior or decision pattern.
- Focus on issues that, if corrected, would materially improve how many top-ranked items users would realistically choose.
"""


class Simulator:
    def __init__(self,llm, model):
        self.llm = llm
        self.user_train = model.user_train
        self.review_data = model.review_data
        self.meta_data = model.meta_data
        api_key = os.environ['OPENAI_API_KEY']
        self.llm_model = OpenAI(api_key=api_key)
        
        self.get_stats()
        self.get_personality()

        self.defined_personality = {
            "activity":{
                0: "LOW: Rarely interacts with the system and does not interact if recommendations are not relevant to their interests.",
                1: "MID: Interacts moderately, primarily when items strictly align with personal preferences.",
                2: "HIGH: Frequently interacts with the system and maintains a high volume of engagement with recommendations."
                },
            "conformity":{
                0: "HIGH: Heavily influenced by popularity and public ratings; tends to follow mainstream trends.",
                1: "MID: Considers both popularity and personal taste, balancing trends with individual preferences.",
                2: "LOW: Ignores popularity and trends, evaluating items purely based on intrinsic personal preference."
                },
            "diversity":{
                0: "LOW: Sticks strictly to a narrow set of familiar categories and avoids exploration.",
                1: "MID: Mostly consumes preferred categories but occasionally explores similar alternatives.",
                2: "HIGH: Seeks high variety and novelty, enjoying the exploration of diverse categories and new styles."
                }}

    def get_stats(self):
        count_dict = defaultdict(int)
        
        total_count = 0
        for v in self.user_train['History'].values():
            for v_ in v:
                count_dict[str(v_)] += 1
                total_count += 1
        
        new_count_dict = {}
        for k, v in count_dict.items():
            new_count_dict[k] = max(v/total_count, 0.0001)
            
        self.item_statistic = new_count_dict
    
    def assign_score(self, value, thresholds):
        if value < thresholds[0]:
            return 0
        elif value < thresholds[1]:
            return 1
        else:
            return 2
    
    def get_personality(self):
        activity_list, div_list,conf_list = [], [],[]
        activity_dict, diversity_dict, conformity_dict = {},{},{}
        for k, v in self.user_train['History'].items():
            activity_list.append(len(v))
            activity_dict[k] = len(v)
            user_cate = []
            user_conf = 0
            count = 0
            for v_ in v:
                user_cate += self.meta_data[str(v_)]['categories']
                try:
                    user_conf += (self.review_data[k][str(v_)] -  self.meta_data[str(v_)]['average_rating'])**2
                    count +=1
                except:
                    0
            if count>0:
                user_conf/=count
            conf_list.append(user_conf)
            div_list.append(len(set(user_cate)))
            diversity_dict[k] = len(set(user_cate))
            conformity_dict[k] = user_conf

        self.personality = {'activity':{}, 'diversity':{}, 'conformity':{}}
        percentiles = {
            'activity': np.percentile(activity_list, [60, 90, 100]),
            'diversity': np.percentile(div_list, [33, 66, 100]),
            'conformity': np.percentile(conf_list, [25, 80, 100])
        }

        for k in self.user_train['History'].keys():
            for dimension, dimension_dict in zip(['activity', 'diversity', 'conformity'],
                                              [activity_dict, diversity_dict, conformity_dict]):
                value = dimension_dict[k]
                self.personality[dimension][k] = self.assign_score(value, percentiles[dimension])
    
    def process_history(self, user_id):
        user_train = []

        for _, hh in enumerate(self.user_train['History'][user_id]):
            # Retrieve item metadata
            m = self.meta_data.get(str(hh), {})
            title = m.get('title', '').strip()
            cats = m.get('categories', [])
            cats_joined = ",".join(cats).strip() if len(cats) >= 2 else ""
            price = str(m.get('price', '')).strip()
            store = str(m.get('store', '')).strip()

            # Handle timestamp
            try:
                ts = int(self.user_train['Time'][user_id][_]) / 1000
                time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            except (IndexError, ValueError):
                time_str = 'No Recorded Time'
            
            # Get user rating
            ratings = 'None'
            if self.review_data.get(user_id, 'None') != 'None':
                ratings_ = self.review_data.get(user_id, 'None').get(str(hh), {}).get('rating', 'None')
                ratings = ratings_ if ratings_ != 'None' else 'None'

            # Append formatted interaction info
            user_train.append(
                f"Interaction Time:{time_str}. "
                f"Title:{title}. "
                f"Category:{cats_joined}. "
                f"Price:{price}. "
                f"Store:{store}. "
                f"Item's popularity (0-1):{self.item_statistic.get(str(hh), 0):.4f}. "
                f"User's Rating:{ratings}\n"
            )

        # Join the list into a single string
        user_train = ''.join(user_train)
        return user_train
                
    def conduct_simulation(self, user_id, rec_results):
        
        rec_page = prompts_list.get_rec_page(rec_results, self.meta_data, self.item_statistic)
        personality_prompt = ""
        for k in self.personality.keys():
            personality_prompt += f"- {k}: {self.defined_personality[k][self.personality[k][str(user_id)]]}\n"
        
        interaction_history = self.process_history(str(user_id))
        
        # Keep this functions to simulate the users -----
        simulate_prompt = prompts_list.perform_recommendation(personality_prompt, rec_page, interaction_history)    
        
        response = self.llm_model.responses.create(model=self.llm,input = simulate_prompt)
        simulate_results = response.output_text
        
        interview_prompt = get_userfeedback(personality_prompt, rec_page, interaction_history, simulate_results)
        response = self.llm_model.responses.create(model=self.llm,input = interview_prompt)
        response_text = response.output_text
        response_text = response_text.replace('**', "")
        # Keep this functions to simulate the users -----
        return response_text