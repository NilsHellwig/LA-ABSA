import ollama
import time
import requests
from openai import OpenAI
import os

class LLM:
    def __init__(self, base_model="gemma2:27b", gwdg_token="", openai_token="",  parameters=[{"name": "stop", "value": ["\\n"]}, 
                                                                                  {"name": "num_ctx", "value": "8192"}]):
        self.model_name = base_model
        self.gwdg_token = gwdg_token
        self.openai_token = openai_token
        
        if self.openai_token != "":
            self.openai_client = OpenAI(api_key=self.openai_token)
        
        
    def predict(self, prompt, seed=0):
        prediction_start_time = time.time()
        if (self.gwdg_token == "" and self.openai_token == ""):
            response = ollama.generate(model=self.model_name, options=dict(seed=seed, temperature=0.8, num_ctx=4096, stop=["]"]), prompt=prompt)
            response = response["response"] + "]"
        if (self.openai_token != ""):
          chat_completion = self.openai_client.chat.completions.create(messages=[{"role": "user", "content": prompt, "stop": ["]"]}],model=self.model_name)
          response = chat_completion.choices[0].message.content
          
          
        if (self.gwdg_token != ""):
          url = "https://chat-ai.academiccloud.de/v1/completions"

          headers = {
           "Accept": "application/json",
           "Authorization": f"Bearer {self.gwdg_token}",
           "Content-Type": "application/json"
          }
          
          data = {
              "model": self.model_name,
              "prompt": prompt,
              "max_tokens": 200,
              "temperature": 0.8,
              "stop": ["]"]
          }
          
          response = requests.post(url, json=data, headers=headers).json()["choices"][0]["text"] + "]"
          
        duration = time.time() - prediction_start_time
        return response, duration