import json
import os
from openai import OpenAI
import pandas as pd
import re
import datetime
from collections import defaultdict
import random


def get_rec_page(items, meta, item_stats):
    return f"""=============    Recommendation Page    =============\n\
== {meta[str(items[0])]['title'].strip()} | History ratings: {meta[str(items[0])]['average_rating']} | Summary: categories-{','.join(meta[str(items[0])]['categories'])}/store-{meta[str(items[0])]['store']}/price-{meta[str(items[0])]['price']}/item's popularity (0-1)-{item_stats[str(items[0])]:.4f}\n\
== {meta[str(items[1])]['title'].strip()} | History ratings: {meta[str(items[1])]['average_rating']} | Summary: categories-{','.join(meta[str(items[1])]['categories'])}/store-{meta[str(items[1])]['store']}/price-{meta[str(items[1])]['price']}/item's popularity (0-1)-{item_stats[str(items[1])]:.4f}\n\
== {meta[str(items[2])]['title'].strip()} | History ratings: {meta[str(items[2])]['average_rating']} | Summary: categories-{','.join(meta[str(items[2])]['categories'])}/store-{meta[str(items[2])]['store']}/price-{meta[str(items[2])]['price']}/item's popularity (0-1)-{item_stats[str(items[2])]:.4f}\n\
== {meta[str(items[3])]['title'].strip()} | History ratings: {meta[str(items[3])]['average_rating']} | Summary: categories-{','.join(meta[str(items[3])]['categories'])}/store-{meta[str(items[3])]['store']}/price-{meta[str(items[3])]['price']}/item's popularity (0-1)-{item_stats[str(items[3])]:.4f}\n\
== {meta[str(items[4])]['title'].strip()} | History ratings: {meta[str(items[4])]['average_rating']} | Summary: categories-{','.join(meta[str(items[4])]['categories'])}/store-{meta[str(items[4])]['store']}/price-{meta[str(items[4])]['price']}/item's popularity (0-1)-{item_stats[str(items[4])]:.4f}\n\
== {meta[str(items[5])]['title'].strip()} | History ratings: {meta[str(items[5])]['average_rating']} | Summary: categories-{','.join(meta[str(items[5])]['categories'])}/store-{meta[str(items[5])]['store']}/price-{meta[str(items[5])]['price']}/item's popularity (0-1)-{item_stats[str(items[5])]:.4f}\n\
== {meta[str(items[6])]['title'].strip()} | History ratings: {meta[str(items[6])]['average_rating']} | Summary: categories-{','.join(meta[str(items[6])]['categories'])}/store-{meta[str(items[6])]['store']}/price-{meta[str(items[6])]['price']}/item's popularity (0-1)-{item_stats[str(items[6])]:.4f}\n\
== {meta[str(items[7])]['title'].strip()} | History ratings: {meta[str(items[7])]['average_rating']} | Summary: categories-{','.join(meta[str(items[7])]['categories'])}/store-{meta[str(items[7])]['store']}/price-{meta[str(items[7])]['price']}/item's popularity (0-1)-{item_stats[str(items[7])]:.4f}\n\
== {meta[str(items[8])]['title'].strip()} | History ratings: {meta[str(items[8])]['average_rating']} | Summary: categories-{','.join(meta[str(items[8])]['categories'])}/store-{meta[str(items[8])]['store']}/price-{meta[str(items[8])]['price']}/item's popularity (0-1)-{item_stats[str(items[8])]:.4f}\n\
== {meta[str(items[9])]['title'].strip()} | History ratings: {meta[str(items[9])]['average_rating']} | Summary: categories-{','.join(meta[str(items[9])]['categories'])}/store-{meta[str(items[9])]['store']}/price-{meta[str(items[9])]['price']}/item's popularity (0-1)-{item_stats[str(items[9])]:.4f}\n\

=============    End Page    =============\n"""


def perform_recommendation(personality, rec_page, interaction_his):
    return f"""
    You excel at role-playing. Picture yourself as a single user exploring a recommendation page like Amazon.
    Your goal is to provide realistic behavior and explain the causal reasons behind your choices to help improve the recommendation model.
    The goal of this task is NOT UI feedback.

    ## Your fixed traits
    {personality}
    
    ## Your Recent context (Recent Interacted Items)
    - {interaction_his}
    
    Below is the current recommendation page from the recommendadtion system:
    
    {rec_page}
    
    ------------------------------------------------------------
    PART 1 — BEHAVIOR (act as a real user)
    For EACH item on the recommendation page, decide:

    1) Whether it aligns with your taste RIGHT NOW (yes or no).
    2) If it aligns and you would choose it this time, give it a rating from 1 to 5 based on how much you think you would like it after trying.
    3) Briefly explain why.

    Use EXACTLY this format for each item (one line per item):

    ItemID: <item_id>; Title: <title>; ALIGN: <yes|no>; RATING: <1-5 or 'NA' if not chosen>; REASON: <short reason>

    After you have done this for all items, DO NOT add any extra commentary in this section.
    """



train_time_prompt = "CRITICAL: The training time per epoch and the validation phase must not exceed 30 minutes. The current code's training time per epoch is estimated to be {time:.1f} minutes. Modify the training code accordingly."
validation_time_prompt = "CRITICAL: The training time per epoch and the validation phase must not exceed 30 minutes. The current code's validating time is estimated to be {time:.1f} minutes. Modify the training code accordingly."


  
def math_diagnosis_agent(raw_diagnosis, model_name):
  api_key = os.environ['OPENAI_API_KEY']
  model = OpenAI(api_key=api_key)
  response = model.responses.create(model=model_name,input = raw_diagnosis)
  final_diagnosis_json = response.output_text
  
  return final_diagnosis_json


def get_status(text):
  pattern = r'"status"\s*:\s*["\']([^"\']+)["\']'
  match = re.search(pattern, text, re.IGNORECASE)
  if match:
        found_status = match.group(1).upper() 
        
        if "CRITICAL" in found_status:
            return True
        else:
            return False
            
  return False

  

def get_summarized_suggestion(total_suggestion, raw_diagnosis, final_diagnosis_json, model_name, prompt):
    total_suggest = ''
    for _, v in enumerate(total_suggestion.values()):
        total_suggest += f"User {_}'s PER_USER_DIAGNOSIS:\n{v}\n\n"
    
    improve_prompt = prompt.format(per_user_json_list=total_suggest)
        
    api_key = os.environ['OPENAI_API_KEY']
    agent = OpenAI(api_key=api_key)
    response = agent.responses.create(model=model_name,input = improve_prompt)
    results_suggestion = response.output_text
      
      
    final_input_for_search_model_simulator = f"""
      [USER DIAGNOSIS]
      {results_suggestion}
      """
    
    final_input_for_search_model_diagnosis = f"""
    [Mathematical Diagnosis]
      - Metrics: {json.dumps(raw_diagnosis['metrics'])}
      - Definitions: {json.dumps(raw_diagnosis['metric_definitions'])}
      - Analysis: {final_diagnosis_json}
    """
    
    if get_status(results_suggestion) and get_status(final_diagnosis_json):
      return True, final_input_for_search_model_simulator, final_input_for_search_model_diagnosis

    return False, final_input_for_search_model_simulator, final_input_for_search_model_diagnosis
  
