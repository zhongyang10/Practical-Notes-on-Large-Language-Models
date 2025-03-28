#构建大模型可调用的函数
import sqlite3

def query_by_product_name(product_name):
    # 连接 SQLite 数据库
    conn = sqlite3.connect('./tools/SportsEquipment.db')
    cursor = conn.cursor()
    # 使用SQL查询按名称查找产品。'%'符号允许部分匹配。
    cursor.execute("SELECT * FROM products WHERE product_name LIKE ?", ('%' + product_name + '%',))
    # 获取所有查询到的数据
    rows = cursor.fetchall()
    # 关闭连接
    conn.close()  
    return rows

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_by_product_name",
            "description": "Query the database to retrieve a list of products that match or contain the specified product name. This function can be used to assist customers in finding products by name via an online platform or customer support interface.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The name of the product to search for. The search is case-insensitive and allows partial matches."
                    }
                },
                "required": ["product_name"]
            }
        }

    }
]



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
        # client = OpenAI(api_key='sk-ebc996d7de55448a856dd4a3fb3ccfc4',base_url='https://api.deepseek.com')
        client = OpenAI()
        result = client.chat.completions.create(
            # model='deepseek-chat',
            model='deepseek-r1:8b',
            messages=self.messages,
            tools= tools
        )
        # return result.choices[0].message.content
        return result
    
    messages = [
    {"role": "user", "content": "老板，在吗"}
]
    
ChatAI = Client()
response = ChatAI("老板，在吗,你们家都卖什么球？")
print(response)