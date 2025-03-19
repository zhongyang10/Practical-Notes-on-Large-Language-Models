from openai import OpenAI
import dspy

class Client:
    def __init__(self, system='你是一个资深教育家'):
        self.system = system
        self.messages = []
        self.sig = Sig
        self.strategyMap = {
            "":dspy.Predict(self.sig),
            "Predict":dspy.Predict(self.sig),
            "ChainOfThought":dspy.ChainOfThought(self.sig),
            "ProgramOfThought":dspy.ProgramOfThought(self.sig)
        }
        self.model = dspy.LM("openai/deepseek-chat", api_key='', api_base='https://api.deepseek.com')
        self._configure_model()
        if system:
            self.messages.append({'role': 'system', 'content': system})

    def __call__(self, message,strategy):
        self.messages.append({'role': 'user', 'content': message})
        try:
            response = self.execute(strategy)
            self.messages.append({'role': 'assistant', 'content': response})#response的格式要根据签名来修改
            return response
        except Exception as e:
            print(f"执行过程中出现错误: {e}")
            return None
        
    def _configure_model(self):
        dspy.configure(lm=self.model)

    def execute(self,strategy):
        try:
            if strategy == "MultiChainComparison":
                classify = dspy.MultiChainComparison([self.chain1,self.chain2])
            elif strategy == "ReAct":
                classify = dspy.ReAct(tools=[])#根据实际业务定义
            else:
                classify = self.strategyMap[strategy]
            response = classify(question = self.messages)
            return response.answer
        except Exception as e:
            print(f"调用 API 时出现错误: {e}")
            return None
        
    def chain1(self,question):
        return dspy.ChainOfThought(self.sig)(question = question).answer
    
    def chain2(self,question):
        return dspy.ChainOfThought(self.sig)(question = question).answer
        
class Sig(dspy.Signature):
    """从多个维度去分析问题给出答案"""
    def __init__(self):
        super().__init__()
    question:str = dspy.InputField()
    answer:str = dspy.OutputField()

# 使用示例
model = Client()
# 调用实例
while True:
    promot = input("\n用户提问：")
    response = model(promot,'ChainOfThought')
    print(response)
