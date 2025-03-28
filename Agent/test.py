# import requests
# from bs4 import BeautifulSoup
# # import serper

# # 使用serper进行搜索
# def serper_search(query):
#     params = {
#         "q": query
#     }
#     headers = {
#         "X-API-KEY": "cb1f6924b3242b0ebe8e1b60d21005f9a370ffef",  # 请替换为你的Serper API Key
#         "Content-Type": "application/json"
#     }
#     response = requests.get("https://google.serper.dev/search", headers=headers, params=params)
#     return response.json()

# # 从网页中提取文字内容
# def extract_text_from_url(url):
#     try:
#         response = requests.get(url)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         return soup.get_text()
#     except:
#         return ""

# # 主函数
# def main():
#     query = input("请输入你的问题: ")
#     search_results = serper_search(query)
#     web_results = search_results.get('organic', [])[:3]  # 取前三个网页结果
#     all_text = ""
#     for result in web_results:
#         url = result.get('link', '')
#         text = extract_text_from_url(url)
#         all_text += text + "\n"
#     print(all_text)

# if __name__ == "__main__":
#     main()


from openai import OpenAI
# 初始化 Ollama 客户端
# client = Client("http://localhost:11434")
client = OpenAI(base_url="http://localhost:11434")
# 定义聊天提示
prompt = "你好，能介绍下你自己吗？"

try:
    # 创建聊天完成
    response = client.chat.completions.create(
        model="deepseek-r1:8b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    # 输出模型的回复
    print(response.choices[0].message.content)
except Exception as e:
    print(f"发生错误: {e}")    
