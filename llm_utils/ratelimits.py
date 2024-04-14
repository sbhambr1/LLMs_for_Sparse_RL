#!/usr/bin/env python
# coding: utf-8


from itertools import count
import openai
import pandas as pd
import tiktoken
import sys
import time
import re
import csv
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


openai.api_key = os.environ["OPENAI_API_KEY"]

def count_tokens(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def truncate_text(text, encoding_name='gpt2', max_tokens=512):
    # Tokenize the text using tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    tokenized_text = encoding.encode(text)
    num_tokens = len(tokenized_text)

    if num_tokens > max_tokens:
        # Calculate the number of tokens to keep
        num_tokens_to_keep = max_tokens - 1  # Subtract 1 to leave space for the truncation token

        # Truncate the tokenized text
        truncated_tokenized_text = {
            'tokens': tokenized_text[:num_tokens_to_keep],
            'is_truncated': True
        }

        # Convert the truncated tokenized text back to a readable string
        truncated_text = encoding.decode(truncated_tokenized_text['tokens'])
        return truncated_text, True
    else:
        return text, False




def generate_set_from_csv(input_string):
    # Split the input string using the comma as the separator
    elements = input_string.split(',')

    # Create a set from the list of elements after stripping whitespaces
    result_set = set(element.strip() for element in elements)

    return result_set

def parse_text_within_tags(input_text):
    # Define the regular expression pattern to match text within "<new>" or "<New>" tags
    pattern = r'<[nN]ew>(.*?)<\/[nN]ew>'
    
    # Find all matches of the pattern in the input text
    matches = re.findall(pattern, input_text)
    
    return matches

def generate_comma_separated_string(input_set):
    # Convert the set elements to strings and join them using the comma separator
    result_string = ', '.join(str(element) for element in input_set)

    return result_string


def get_opposite_label(pred_label, task):
    if task=='snli':
        opp_map = {'contradiction':'entailment', 'entailment':'contradiction', 'neutral':'contradiction'}
        return opp_map[pred_label]
    elif task=='imdb':
        ## 0 is negative, 1 is positive
        opp_map = {'positive': 'negative', 'negative':'positive', 0.0:1.0, 1.0:0.0}
        return opp_map[pred_label]
    elif task=='ag_news':
        opp_map = {'the world': 'business', 'business':'sports', 'sports':'the world', 'science/tech': 'sports'}
        return opp_map[pred_label]

def get_opposit_set_of_labels(pred_label, task):
    if task=='snli':
        opp_set_map = {
            'entailment': "'contradiction' or 'neutral'",
            'contradiction': "'entailment' or 'neutral'",
            'neutral': "'contradiction' or 'entailment'"
        }
        return opp_set_map[pred_label]
    elif task=='ag_news':
        opp_set_map = {
            'the world': "'business', 'sports' or 'science/tech'",
            'business': "'the world', 'sports' or 'science/tech'",
            'sports': "'business', 'the world' or 'science/tech'",
            'science/tech': "'the world', 'sports' or 'business'"
        }
        return opp_set_map[pred_label]
    else:
        raise ValueError('invalid value for task')

file_path_map = {
    'distilbert-snli': "/home/local/ASUAD/abhatt43/Projects/chatgpt-explanation/all-tasks/DistilBERT/snli/distilbert-snli-triples.csv",
    'distilbert-imdb': "/home/local/ASUAD/abhatt43/Projects/chatgpt-explanation/all-tasks/DistilBERT/imdb/distilbert-imdb-triples.csv",
    'lstm-imdb': "/home/local/ASUAD/abhatt43/Projects/chatgpt-explanation/all-tasks/RNN/lstm-imdb-triples.csv",
    'distilbert-ag_news': "/home/local/ASUAD/abhatt43/Projects/chatgpt-explanation/all-tasks/DistilBERT/ag_news/distilbert-ag_news-triples.csv"
}

## change as needed

test_type = 'distilbert-ag_news'
task = 'ag_news'

all_data = []
correctly = []
with open(file_path_map[test_type], 'r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Skip the header (first row)
    next(csv_reader)

    # Iterate over the rows and append data to the list
    for row in csv_reader:
        all_data.append(row)
        if row[1]==row[2]:
            correctly.append(row)

print("all data: ", len(all_data), ", correctly: ", len(correctly), ", accuracy: ", float(len(correctly))/len(all_data))



## For GPT-4, also need count of tokens in a 1 min window. Should be <= 10,000

tokens_per_min = 0

@retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(6))
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


# ## Generate counterfactual explanations for only correctly classified samples:
# model_engine = "gpt-3.5-turbo-16k"
model_engine = 'gpt-4'
# max_tokens = 1024
MAX_TOKENS_CONTEXT = 8192

# this is for multi class classification tasks such as SNLI, AG_NEWS
# 'any_of': for a given label, ask LLM to flip to any of the other labels 
# 'strict': for a given label, use the hardcoded opposite label in opposite_label dict
cf_mode = 'any_of'  ## one of 'any_of', 'strict'
cf_explanations = []
parsing_fail = 0

column_names = ["original_text", "ground_truth", "y_pred_original"]
out_df = pd.DataFrame(correctly[:100], columns=column_names)

for i, instance in enumerate(correctly[:100]):

    ## build initial prompt to perform feature selection
    
    text = instance[0]
    gt = instance[1]
    pred = instance[2]

    if task=='imdb' or task=='ag_news':
        truncated_text, is_truncated = truncate_text(text, encoding_name='cl100k_base', max_tokens=128)
        text = truncated_text

    if task=='snli':
        initial_prompt = "You are an oracle explanation module in a machine learning pipeline. In the task of natural language inference on the SNLI dataset, a trained black-box classifier correctly predicted the label '"+pred+"' for the following text. Explain why the model predicted the '"+pred+"' label by identifying the latent features that caused the label. List ONLY the latent features as a comma separated list. Examples of latent features are 'tone', 'ambiguity in text', etc.\n---\nText: \""+text+"\"\n---\nBegin!"
    elif task=='imdb':
        initial_prompt = "You are an oracle explanation module in a machine learning pipeline. In the task of sentiment analysis on the IMDB dataset, a trained black-box classifier correctly predicted the label '"+pred+"' for the following text. Explain why the model predicted the '"+pred+"' label by identifying the latent features that caused the label. List ONLY the latent features as a comma separated list. Examples of latent features are 'tone', 'ambiguity in text', etc.\n---\nText: \""+text+"\"\n---\nBegin!"
    elif task=='ag_news':
        initial_prompt = "You are an oracle explanation module in a machine learning pipeline. In the task of news topic classification on the AG News dataset, a trained black-box classifier correctly predicted the label '"+pred+"' for the following text. Explain why the model predicted the '"+pred+"' label by identifying the latent features that caused the label. List ONLY the latent features as a comma separated list. Examples of latent features are 'tone', 'ambiguity in text', etc.\n---\nText: \""+text+"\"\n---\nBegin!"

    max_tokens = 256

    tokens_per_min += count_tokens(initial_prompt,  'cl100k_base') + max_tokens
    if tokens_per_min > 10000:
        time.sleep(55)
        tokens_per_min = 0

    completion = chat_completion_with_backoff(
        model=model_engine,
        messages=[
            {'role': 'user', 
             'content': initial_prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.4,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    response = completion.choices[0].message.content

    if tokens_per_min==0:
        tokens_per_min += count_tokens(initial_prompt,  'cl100k_base') + max_tokens
    if 'latent features:' in response:
        latent_features = response.split(':')[1]
    latent_features = generate_set_from_csv(response)

#     print(latent_features)
    

    word_set = set()
    for feature in latent_features:
        # time.sleep(1.5)
        prompt5 = "Identify the words in the text that is associated with the latent feature \""+feature+"\" and output as a comma separated list."
        messages=[
            {
                'role': 'user',
                'content': initial_prompt
            },
            {
                'role': 'assistant',
                'content': response
            },
            {
                'role': 'user',
                'content': prompt5
            }
        ]
        context = initial_prompt + response + prompt5
        context_token_count = count_tokens(context, 'cl100k_base')

        # max_tokens = MAX_TOKENS_CONTEXT - context_token_count - 512
        max_tokens = 128

        tokens_per_min += context_token_count + max_tokens

        if tokens_per_min > 10000:
            time.sleep(55)
            tokens_per_min = 0

        try:
            completion = chat_completion_with_backoff(
            model=model_engine,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.4,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
            if tokens_per_min==0:
                tokens_per_min += context_token_count + max_tokens
        except Exception as e:
            print("Exception: ", e)
            print("CF explanations so far: ")
            print(cf_explanations)
            exit(0)
        words = completion.choices[0].message.content
        word_set = word_set | generate_set_from_csv(words)

    all_words = generate_comma_separated_string(word_set)
    ## now to generate the cf explanation

    prompt6 = "Identify all words that are associated with the latent features you identified."

    opposite_label = get_opposite_label(pred, task=task)

    task_desc = {
        'snli': "natural language inference",
        'imdb': "sentiment classification",
        'ag_news': "news topic classification"
    }

    if cf_mode=='strict':
        opposite_label = get_opposite_label(pred, task=task)
        prompt7 = "Generate a counterfactual explanation for the original text by ONLY changing a minimal set of the words you identified. It is okay if the semantic meaning of the original text is altered. Remember that the task is "+task_desc[task]+", and you have to change the label from '"+pred+"' to '"+opposite_label+"'. Use the following definition of 'counterfactual explanation': \"A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome.\" Enclose the generated text within \"<new>\" tags."
    elif cf_mode=='any_of':
        opposite_label_set = get_opposit_set_of_labels(pred, task=task)
        prompt7 = "Generate a counterfactual explanation for the original text by ONLY changing a minimal set of the words you identified. It is okay if the semantic meaning of the original text is altered. Remember that the task is "+task_desc[task]+", and you have to change the label from '"+pred+"' to any of "+opposite_label_set+". Use the following definition of 'counterfactual explanation': \"A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome.\" Enclose the generated text within \"<new>\" tags."



    if model_engine == 'gpt-4':
        if task == 'snli':
            shorter_initial_prompt = "You are an oracle explanation module in a machine learning pipeline. In the task of natural language inference on the SNLI dataset, a trained black-box classifier correctly predicted the label '"+pred+"' for the following text.\n---\nText: \""+text+"\""
        elif task == 'imdb':
            shorter_initial_prompt = "You are an oracle explanation module in a machine learning pipeline. In the task of sentiment analysis on the IMDB dataset, a trained black-box classifier correctly predicted the label '"+pred+"' for the following text.\n---\nText: \""+text+"\""
        elif task=='ag_news':
            shorter_initial_prompt = "You are an oracle explanation module in a machine learning pipeline. In the task of news topic classification on the AG News dataset, a trained black-box classifier correctly predicted the label '"+pred+"' for the following text.\n---\nText: \""+text+"\""

       
        messages = [
        {
            'role': 'user',
            'content': shorter_initial_prompt
        },
        # {
        #     'role': 'assistant',
        #     'content': response
        # },
        # {
        #     'role': 'user',
        #     'content': prompt6
        # },
        {
            'role': 'assistant',
            'content': 'Words identified that "causes" the label \''+pred+'\': ' + all_words
        },
        {
            'role': 'user',
            'content': prompt7
        }
    ]

    context = shorter_initial_prompt + 'Words identified that "causes" the label \''+pred+'\': ' + all_words + prompt7
    context_token_count = count_tokens(context, 'cl100k_base')

    # max_tokens = MAX_TOKENS_CONTEXT - context_token_count - 512
    max_tokens = 256

    tokens_per_min += context_token_count + max_tokens
    if tokens_per_min > 10000:
        time.sleep(55)
        tokens_per_min=0
    
    # time.sleep(3)
    completion = chat_completion_with_backoff(
        model=model_engine,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.4,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    if tokens_per_min==0:
        tokens_per_min += context_token_count + max_tokens

    cf_response = completion.choices[0].message.content
    if parse_text_within_tags(cf_response)==[]:
        print(i, cf_response)
        cf_explanations.append('null')
        parsing_fail+=1
    else:
        cf_explanations.append(parse_text_within_tags(cf_response)[0].strip())
    

    
    
print('Done!')
print('Parsing Fail Count: ', parsing_fail)
try:
    out_df['counterfactual_text'] = cf_explanations


    if cf_mode=='any_of':
        out_df.to_csv("any_of-gpt4-" + test_type + "-cf-explanations.csv", index=False)
    else:
        out_df.to_csv("gpt4-" + test_type + "-cf-explanations.csv", index=False)
except Exception as e:
    print(cf_explanations)
