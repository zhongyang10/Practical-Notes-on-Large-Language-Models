from openai import OpenAI
from pydantic import BaseModel

class Client:
    def __init__(self,system = ''):
        self.system = system
        self.messages = []
        if system:
            self.messages.append({'role':'system','content':system})

    def __call__(self, message):
        self.messages.append({'role':'user','content':message})
        response = self.execute1()
        self.messages.append({'role':'assistant','content':response})
        return response
    
    def execute1(self):
        client = OpenAI(api_key='',base_url='https://api.deepseek.com')
        result = client.chat.completions.create(
            model='deepseek-chat',
            messages=self.messages
        )
        return result.choices[0].message.content


