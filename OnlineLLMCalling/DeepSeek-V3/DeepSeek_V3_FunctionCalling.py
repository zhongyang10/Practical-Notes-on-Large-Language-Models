#单次函数调用

from openai import OpenAI
import json

def get_weather(location):
    return location + """最近的天气以多云和晴天为主，气温波动较大，且伴有大风和沙尘天气。以下是具体情况：
- **今日：多云，有浮尘，山区有雨夹雪，北风一二级转三四级，最高温度17℃，最低温度9℃。16时气温17℃，南风1级，湿度25；傍晚起风力逐渐加大，夜间多云，有沙尘，北风四级左右，阵风七八级。
- **未来7天**：
    - 第1天：多云，气温3℃～14℃，西北风3级，湿度15。
    - 第2天：多云，1℃～10℃，西北风4级，湿度14。
    - 第3天：晴，2℃～11℃，西北风4级，湿度12。
    - 第4天：晴，3℃～15℃，南风3级，湿度13。
    - 第5天：晴，5℃～19℃，西北风2级，湿度15。
    - 第6天：晴，6℃～20℃，西北风2级，湿度15。
    - 第7天：晴，10℃～24℃，南风2级，湿度16。"""

get_weather_function={
    'name':'get_weather',
    'description':'查询即时天气函数，根据输入的城市名称，查询对应城市的实时天气',
    'parameters':{
        'type':'object',
        'properties':{
            'location':{
                'description':"城市名称，注意，中国的城市需要用对应城市的英文名称代替，例如如果需要查询北京市天气，则loc参数需要输入'北京'",
                'type':'string'
            }
        },
        'required':['location']
    }
}

tools = [
    {
        "type":"function",
        "function":get_weather_function
    }
]

def run_conv(messages, 
             api_key,
             tools=None, 
             functions_list=None,
             model="deepseek-chat"):
    """
    能够自动执行外部函数调用的Chat对话模型
    :param messages: 必要参数，输入到Chat模型的messages参数对象
    :param api_key: 必要参数，调用模型的API-KEY
    :param tools: 可选参数，默认为None，可以设置为包含全部外部函数的列表对象
    :param model: Chat模型，可选参数，默认模型为deepseek-chat
    :return：Chat模型输出结果
    """
    user_messages = messages
    
    
    client = OpenAI(api_key=api_key, 
                base_url="https://api.deepseek.com")
    
    # 如果没有外部函数库，则执行普通的对话任务
    if tools == None:
        response = client.chat.completions.create(
            model=model,  
            messages=user_messages
        )
        final_response = response.choices[0].message.content
        
    # 若存在外部函数库，则需要灵活选取外部函数并进行回答
    else:
        # 创建外部函数库字典
        available_functions = {func.__name__: func for func in functions_list}

        # 创建包含用户问题的message
        messages = user_messages
        
        # first response
        response = client.chat.completions.create(
            model=model,  
            messages=user_messages,
            tools=tools,
        )
        response_message = response.choices[0].message
        # print("response_message：",response_message)
        # print("esponse_message.model_dump()：",response_message.model_dump())

        # 获取函数名
        function_name = response_message.tool_calls[0].function.name
        # 获取函数对象
        fuction_to_call = available_functions[function_name]
        # 获取函数参数
        function_args = json.loads(response_message.tool_calls[0].function.arguments)
        
        # 将函数参数输入到函数中，获取函数计算结果
        function_response = fuction_to_call(**function_args)

        # messages中拼接first response消息
        user_messages.append(response_message.model_dump())  
        
        # messages中拼接外部函数输出结果
        user_messages.append(
            {
                "role": "tool",
                "content": function_response,
                "tool_call_id":response_message.tool_calls[0].id
            }
        )
        
        # 第二次调用模型
        second_response = client.chat.completions.create(
            model=model,
            messages=user_messages)
            
        # 获取最终结果
        final_response = second_response.choices[0].message.content
    
    return final_response

messages = [{"role": "user", "content": "请问北京最近天气如何？给一些出行建议"}]
ds_api_key = 'sk-ebc996d7de55448a856dd4a3fb3ccfc4'

final_response = run_conv(messages=messages, 
         api_key = ds_api_key,
         tools=tools, 
         functions_list=[get_weather])

print(final_response)