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
        response = self.execute()
        self.messages.append({'role':'assistant','content':response})
        return response
    
    def execute(self):
        client = OpenAI(api_key='',base_url='https://api.deepseek.com')
        result = client.chat.completions.create(
            model='deepseek-chat',
            messages=self.messages
        )
        return result.choices[0].message.content


ChatAI = Client()
#zero shot：指模型在没有见过任何特定任务示例的情况下，直接根据其预训练知识进行推理和预测的能力。
prompt = "将[上海自来水来自海上]反过来写"
response1 = ChatAI(prompt)
print(f"response1:{response1}")

#few shot: 指模型在只看到少量特定任务示例的情况下，就能快速学习并适应新任务的能力。
prompt = "将[上海自来水来自海上]反过来写,例如：xyz，反过来则是：zyx"
response2 = ChatAI(prompt)
print(f"response2:{response2}")

#COT：让prompt包含了一些思路示例。它与n-shot提示技术不同，因为思维链提示的结构是为了引导模型具备批判性思维并帮助推理思考，让它LLMs发现可能没有考虑到的新方法,这里也是和agent接轨的地方。
prompt = "将[上海自来水来自海上呢]反过来写,Think carefully and logically, explaining your answer."
response3 = ChatAI(prompt)
print(f"response3:{response3}")

#TOT、GOT

