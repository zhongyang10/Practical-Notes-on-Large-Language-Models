from openai import OpenAI

class ChatBot:
    """
    优化ChatBot的代码结构：
    定义一个ChatBot的类，用来创建和处理一个基于DeepSeek模型的聊天机器人;
    init方法用来接收系统提示(System Prompt)，并追加到全局的消息列表中，它是类的构造函数，当创建类的实例时会自动调用。self是类实例的引用，system是一个可选参数，默认值为空字符串。
    call方法是Python类的一个特殊方法,当对一个类的实例像调用函数一样传递参数并执行时，实际上就是在调用这个类的call方法。其内部会调用execute方法;
    execute 方法实际上就是与DeepSeek的API进行交互，发送累积的消息历史（包括系统消息、用户消息和之前的回应）到DeepSeek的聊天模型,返回最终的响应。
    """
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
    
    def execute(self):
        client = OpenAI(api_key='',base_url='https://api.deepseek.com')
        completion = client.chat.completions.create(model="deepseek-chat", messages=self.messages)
        return completion.choices[0].message.content
    
chatBot1 = ChatBot()
while True:
    message = input("\n用户输入：")
    if message == "退出":
        break
    response = chatBot1(message)
    print(response)
