from openai import OpenAI

#创建客户端,api_key需要在deepseek官网注册获取
client = OpenAI(api_key='',base_url='https://api.deepseek.com')

#通过system定义客户端的角色，这里可以告诉系统他要以一个什么角色来回答问题
systemPrompt = "你是一个经验丰富的医生。"
messages = []
messages.append({
    'role':'system',
    'content':systemPrompt
})

while True:
    userPrompt = input("\n用户提问: ")
    if userPrompt == "退出":
        break
    messages.append({
        'role':'user',
        'content':userPrompt
    })
    response = client.chat.completions.create(
        model='deepseek-chat',
        messages=messages
    )
    print(f'model response:{response.choices[0].message.content}')
    #把模型回复加入历史对话，使模型具有记忆功能
    messages.append({
        'role':'assistant',
        'content':response.choices[0].message.content
    })

