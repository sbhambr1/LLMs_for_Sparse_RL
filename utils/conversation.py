import os
from openai import OpenAI
import pickle as pkl 
from ratelimit import limits, sleep_and_retry

api_key = os.environ["OPENAI_API_KEY"]

if api_key is None:
    raise Exception("Please insert your OpenAI API key in conversation.py")

client = OpenAI(
  api_key=api_key,
)

class Conversation:
    def __init__(self, llm_actor_model, llm_human_proxy_model) -> None:
        self.llm_actor_prompt = []
        self.llm_human_proxy_prompt = []
        self.log_history_actor = []
        self.log_history_human_proxy = []
        self.llm_actor_model =  llm_actor_model
        self.llm_human_proxy_model = llm_human_proxy_model

    # avoid using for now        
    def construct_message(self, prompt, role, use_context, model):
        assert role in ["user", "assistant"]

        new_message = {"role": role, "content": prompt}
        if use_context:
            if model == "llm_actor":
                message = self.llm_actor_prompt + [new_message]
            else:
                message = self.llm_human_proxy_prompt + [new_message]
        else:
            message = [new_message]
        return message
    
    def llm_human_proxy(self, prompt, stop, temperature=0, role="user", use_context=False):    
        # chat model
        
        message = self.construct_message(prompt, role, use_context, "llm_human_proxy")  
        
        response = client.chat.completions.create(
        model=self.llm_human_proxy_model,
        messages = message,
        temperature=temperature,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop
        )

        answer = response.choices[0].message.content
        self.log_history_human_proxy.append(answer)
        self.llm_human_proxy_prompt.append(prompt + answer + "\n")
        return answer

    def llm_actor(self, prompt, stop, temperature=0, role="user", use_context=False): 
        # chat model       
        
        message = self.construct_message(prompt, role, use_context, "llm_actor")  
        
        response = client.chat.completions.create(
        model=self.llm_actor_model,
        messages = message,
        temperature=temperature,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop
        )

        answer = response.choices[0].message.content
        self.log_history_actor.append(answer)
        self.llm_actor_prompt.append(prompt + answer + "\n")
        return answer

    def add_prompt(self, prompt, role):
        raise NotImplementedError
        if isinstance(prompt, client.OpenAIObject): #TODO: check this api object
            message = {"role" : role, "content": prompt['choices'][0]['message']['content']}
        else : 
            message = {"role" : role, "content": prompt}
        self.prompt.append(message)
    
    def save(self, filename):
        raise NotImplementedError
        with open(filename, 'wb') as f:
            object = {"prompt": self.prompt, "log_history": self.log_history, "model": self.model}
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