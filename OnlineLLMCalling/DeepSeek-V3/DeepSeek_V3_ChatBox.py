# #deepseek-v3调用风格和OpenAI完全一致，Function calling、提示词缓存、Json Output等功能完全相同
from openai import OpenAI
import re
import json

class ChatBox:
    def __init__(self, model_name, model_api_key, model_api_base,system = ''):
        self.client = OpenAI(api_key=model_api_key,base_url=model_api_base)
        self.messages = []
        self.model_name = model_name
        if system:
            self.messages.append({"role":"system","content":system})

    def __call__(self, message):
        try:
            response = self.execute(message)
            return response
        except Exception as e:
            print(f"执行过程中出现错误: {e}")
            return None
        
    def execute(self,message):
        dic_message = self.process_user_input(message)
        self.messages.append(dic_message)
        response = self.chat_with_deepseek()
        self.messages.append({"role":"assistant","content":response})
        return response

    def extract_url_and_text(self,input_text):
        """
        为适配deepseek未来的多模态功能,此处先实现提取用户输入的URL和描述性文本
        参数：
        input_text(str):用户输入的文本,可能包含URL和描述性文本
        返回：
        tuple,包含描述性文本和提取到的URL,如果没有URL,则返回(input_text,None)
        """
        #使用正则表达式检测URL
        url_pattern = re.compile(r'(https?://[^\s]+)')
        url_match = url_pattern.search(input_text)

        if url_match:
            url = url_match.group(0)
            description = input_text.replace(url,'').strip()
            return description,url
        else:
            return input_text,None
        
    def creat_message(self,role,content):
        return {"role":role,"content":content}
    
    def create_user_nmessage_with_image(self,text,image_url):
        return {
            "role":"user",
            "content":[
                {"type":"text","text":text},
                {"type":"image_url","image_url":{"url",image_url}}
            ]
        }

    
    def process_user_input(self,input_text):
        """
        根据用户输入信息判断是否包含URL并生成对应的消息格式
        参数：
        input_text:用户的输入文本
        返回：
        dict:生成的用户消息包含文本或图片url
        """
        description,url = self.extract_url_and_text(input_text)
        if url:
            if not description:
                description = "请帮我分析这张图片的内容."
            return self.create_user_nmessage_with_image(description,url)
        else:
            return self.creat_message("user",description)
        
    def chat_with_deepseek(self):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages = self.messages
        )
        return response.choices[0].message.content
    
if __name__ == "__main__":
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("错误: 未找到 config.json 文件。")
    except json.JSONDecodeError:
        print("错误: 无法解析 config.json 文件。")
    else:
        model_name = config['model_name']
        model_api_key = config['model_api_key']
        model_api_base = config['model_api_base']
        system_prompt = config['system_prompt']
        model = ChatBox(model_name, model_api_key, model_api_base,system_prompt)

        while True:
            query = input("输入（输入“退出”可退出对话）：")
            if query == "退出":
                break
            response = model(query)
            print(response)