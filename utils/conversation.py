import os
import tiktoken
from openai import OpenAI
import pickle as pkl 
from ratelimit import limits, sleep_and_retry

api_key = os.environ["OPENAI_API_KEY"]
# key_file = open(os.getcwd()+'/key.txt', 'r')
# api_key = key_file.readline().rstrip()

if api_key is None:
    raise Exception("Please insert your OpenAI API key in conversation.py")

client = OpenAI(
  api_key=api_key,
)

class Conversation:
    def __init__(self, llm_model) -> None:
        self.llm_prompt = []
        self.log_history = []
        self.llm_model =  llm_model
        self.tokens_per_min = 0
        self.max_tokens = 256
        self.input_token_cost = 0.5 / 1e6
        self.output_token_cost = 1.5 / 1e6
        self.total_cost = 0
        self.cost_limit = 10
        
    def count_tokens(self, string: str, encoding_name: str) -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
       
    def construct_message(self, prompt, role):
        assert role in ["user", "assistant"]
        new_message = {"role": role, "content": prompt}
        self.llm_prompt = []
        message = self.llm_prompt + [new_message]
        # input_tokens = self.count_tokens(message[0]['content'],  'cl100k_base')
        # self.total_cost += self.input_token_cost * input_tokens
        return message

    def llm_actor(self, prompt, stop, temperature=0, role="user"): 
        # chat model       
        
        message = self.construct_message(prompt, role)  
        
        if self.total_cost > self.cost_limit:
            return {"response_message": "[WARNING] COST LIMIT REACHED!"}
        else:
            response = client.chat.completions.create(
            model=self.llm_model,
            messages = message,
            temperature=temperature,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop
            )

        answer = response.choices[0].message.content
        # output_tokens = self.count_tokens(answer, 'cl100k_base')
        # self.total_cost += self.output_token_cost * output_tokens
        self.log_history.append(answer)
        self.llm_prompt.append(prompt + answer + "\n")
        return answer
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            object = {"prompt": self.llm_prompt, "log_history": self.log_history, "model": self.llm_model}
            pkl.dump(object, f)


if __name__ == "__main__":
    c = Conversation("gpt-3.5-turbo")
    PROMPT_1 = "Hello, how are you?"
    PROMPT_2 = "I've been better."

    print(f"User : {PROMPT_1}")
    x1 = c.get_response(PROMPT_1)
    print("LLM : ", x1['response_message'])
    
    print(f"User : {PROMPT_2}")
    x2 = c.get_response(PROMPT_2)
    print("LLM : ", x2['response_message'])

    print("done")