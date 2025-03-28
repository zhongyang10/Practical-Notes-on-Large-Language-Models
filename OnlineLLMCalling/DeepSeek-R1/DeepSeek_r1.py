from openai import OpenAI
import os
from IPython.display import display, Code, Markdown
import requests
import json

client = OpenAI(api_key='',base_url='https://api.deepseek.com')

prompt = """
我想构建一个Python应用程序，它能够接收用户的问题，并在数据库中查找与之对应的答案。
如果有接近匹配的答案，它就检索出匹配的答案。
如果没有，它就请求用户提供答案，并将问题/答案对存储在数据库中。
为我规划所需的目录结构，然后完整地返回每个文件。只需在代码的开始和结束时提供你的推理，而不用贯穿整个代码。
"""

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {
            "role":"user",
            "content":[
                {
                    "type":"text",
                    "text":prompt
                }
            ]
        }
    ]
)

# 将内容保存到本地文件
file_path = "response_markdown.md"
try:
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(response.choices[0].message.content)
    print(f"Markdown 内容已成功保存到 {file_path}")
except Exception as e:
    print(f"保存文件时出现错误: {e}")