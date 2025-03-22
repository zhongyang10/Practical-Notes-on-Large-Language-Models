import requests
import json
from pprint import pprint
import hashlib
from langchain.docstore.document import Document
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from html2text import HTML2Text
import re

import dspy


class SearchResultProcessor:
    def __init__(self, api_key, query, search_params):
        """
        初始化 SearchResultProcessor 类的实例。

        :param api_key: 用于验证请求的 API 密钥
        :param query: 要搜索的问题
        """
        self.api_key = api_key
        self.query = query
        self.params = {
            'api_key': self.api_key,
            'q': self.query,
            **search_params
        }

    def md5(self, data: str):
        """
        计算输入字符串的 MD5 哈希值。

        :param data: 输入的字符串
        :return: 计算得到的 MD5 哈希值
        """
        try:
            _md5 = hashlib.md5()
            _md5.update(data.encode("utf-8"))
            _hash = _md5.hexdigest()
            return _hash
        except UnicodeEncodeError:
            print("编码错误: 无法对输入字符串进行 UTF-8 编码。")
            return None

    def fetch_search_results(self, api_url):
        """
        发起 GET 请求到 Serper API 并获取搜索结果。

        :return: 包含搜索结果的列表
        """
        try:
            api_result = requests.get(api_url, params=self.params)
            api_result.raise_for_status()
            search_data = api_result.json()
            items = search_data.get("organic", [])
            results = []
            for item in items:
                # 为每个搜索结果生成 UUID（MD5 哈希）
                item["uuid"] = hashlib.md5(item["link"].encode()).hexdigest()
                # 初始化搜索结果的得分
                item["score"] = 0.00
                results.append(item)
            return results
        except requests.RequestException as e:
            print(f"请求错误: {e}")
            return []
        except ValueError as e:
            print(f"JSON 解析错误: {e}")
            return []

    def create_documents(self, results):
        """
        根据搜索结果创建 Document 对象列表。

        :param results: 搜索结果列表
        :return: Document 对象列表
        """
        documents = []
        for result in results:
            try:
                if "uuid" in result:
                    uuid = result["uuid"]
                else:
                    uuid = self.md5(result["link"])
                text = result["snippet"]
                document = Document(
                    page_content=text,
                    metadata={
                        "uuid": uuid,
                        "title": result["title"],
                        "snippet": result["snippet"],
                        "link": result["link"],
                    },
                )
                documents.append(document)
            except KeyError as e:
                print(f"键错误: 搜索结果中缺少必要的键 {e}。")
        return documents

    def calculate_similarity_scores(self, documents):
        """
        计算每个文档与查询的相似度得分，并对文档进行排序。

        :param documents: Document 对象列表
        :return: 按相似度得分排序后的前三个 Document 对象列表
        """
        try:
            normal = NormalizedLevenshtein()
            for x in documents:
                # 对于每个文档，计算查询与文档内容的相似度
                x.metadata["score"] = normal.similarity(self.query, x.page_content)
            documents.sort(key=lambda x: x.metadata["score"], reverse=True)
            return documents[:2]
        except Exception as e:
            print(f"相似度计算错误: {e}")
            return []

    def fetch_html_content(self, documents):
        """
        从文档的链接中获取 HTML 内容，并将其转换为 Markdown 格式。

        :param documents: Document 对象列表
        :return: 包含链接和对应 Markdown 内容的字典
        """
        url_list = [document.metadata['link'] for document in documents if 'link' in document.metadata]
        html_response = []
        for url in url_list:
            try:
                html = requests.get(url)
                html.raise_for_status()
                response = html.text
                html_response.append(response)
            except requests.RequestException as e:
                print(f"请求错误: 无法获取 {url} 的内容。错误信息: {e}")
        markdown_response = []
        for html in html_response:
            try:
                converter = HTML2Text()
                converter.ignore_links = True
                converter.ignore_images = True
                markdown = converter.handle(html)
                # 美化 Markdown 文本，去除多余换行符
                markdown = re.sub(r'\n{3,}', '\n\n', markdown)
                markdown_response.append(markdown)
            except Exception as e:
                print(f"HTML 转 Markdown 错误: {e}")
        combined_list = list(zip(url_list, markdown_response))
        content_maps = {}
        for url, content in combined_list:
            content_maps[url] = content
        return content_maps

    def clean_text(self, text):
        """
        清洗文本，去除除文字、数据以外的内容
        :param text: 待清洗的文本
        :return: 清洗后的文本
        """
        try:
            # 去除 HTML 标签
            text = re.sub(r'<[^>]+>', '', text)
            # 去除非中文、英文、数字、空格和换行符的字符
            cleaned_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\n]', '', text)
            # 去除多余的空格和换行符
            cleaned_text = re.sub(r' +', ' ', cleaned_text)
            cleaned_text = re.sub(r'\n+', ' ', cleaned_text).strip()
            return cleaned_text
        except Exception as e:
            print(f"文本清洗错误: {e}")
            return text

    def update_documents_with_content(self, documents, content_maps):
        """
        更新文档的元数据，添加对应的 Markdown 内容，并清洗文本。

        :param documents: Document 对象列表
        :param content_maps: 包含链接和对应 Markdown 内容的字典
        :return: 更新后的 Document 对象列表
        """
        for result in documents:
            try:
                if result.metadata['link'] in content_maps:
                    content = content_maps[result.metadata['link']]
                    cleaned_content = self.clean_text(content)
                    result.metadata['content'] = cleaned_content
            except KeyError as e:
                print(f"键错误: 文档元数据中缺少必要的键 {e}。")
        return documents

    def process(self, api_url):
        """
        处理搜索结果的主方法，依次调用各个步骤。

        :return: 处理后的 Document 对象列表
        """
        results = self.fetch_search_results(api_url)
        documents = self.create_documents(results)
        top_documents = self.calculate_similarity_scores(documents)
        content_maps = self.fetch_html_content(top_documents)
        updated_documents = self.update_documents_with_content(top_documents, content_maps)
        return updated_documents


class Client:
    def __init__(self, system, model_name, model_api_key, model_api_base):
        self.system = system
        self.messages = []
        self.sig = Sig2
        self.strategyMap = {
            "": dspy.Predict(self.sig),
            "Predict": dspy.Predict(self.sig),
            "ChainOfThought": dspy.ChainOfThought(self.sig),
            "ProgramOfThought": dspy.ProgramOfThought(self.sig)
        }
        try:
            self.model = dspy.LM(model_name, api_key=model_api_key, api_base=model_api_base)
            self._configure_model()
            if system:
                self.messages.append({'role': 'system', 'content': system})
        except Exception as e:
            print(f"模型初始化错误: {e}")

    def __call__(self, message, strategy):
        self.messages.append({'role': 'user', 'content': message})
        try:
            response = self.execute(strategy)
            self.messages.append({'role': 'assistant', 'content': response})
            return response
        except Exception as e:
            print(f"执行过程中出现错误: {e}")
            return None

    def _configure_model(self):
        try:
            dspy.configure(lm=self.model)
        except Exception as e:
            print(f"模型配置错误: {e}")

    def execute(self, strategy):
        try:
            if strategy == "MultiChainComparison":
                classify = dspy.MultiChainComparison([self.chain1, self.chain2])
            elif strategy == "ReAct":
                classify = dspy.ReAct(tools=[])
            else:
                classify = self.strategyMap[strategy]
            response = classify(question=self.messages)
            return response.answer
        except Exception as e:
            print(f"调用 API 时出现错误: {e}")
            return None

    def chain1(self, question):
        try:
            return dspy.ChainOfThought(self.sig)(question=question).answer
        except Exception as e:
            print(f"Chain1 执行错误: {e}")
            return None

    def chain2(self, question):
        try:
            return dspy.ChainOfThought(self.sig)(question=question).answer
        except Exception as e:
            print(f"Chain2 执行错误: {e}")
            return None


class Sig(dspy.Signature):
    """结合输入信息，从多个维度去分析问题给出答案"""
    def __init__(self):
        super().__init__()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Sig2(dspy.Signature):
    """从多个维度去分析问题给出答案"""
    def __init__(self):
        super().__init__()

    question: list[str] = dspy.InputField(desc="输入的文本，你需要将该文本的主要信息提取出来")
    answer: str = dspy.OutputField(desc="这是你根据question提取的主要信息，用中文回答，最多500字")


if __name__ == "__main__":
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("错误: 未找到 config.json 文件。")
    except json.JSONDecodeError:
        print("错误: 无法解析 config.json 文件。")
    else:
        api_key = config['api_key']
        serper_api_url = config['serper_api_url']
        search_params = config['search_params']
        model_name = config['model_name']
        model_api_key = config['model_api_key']
        model_api_base = config['model_api_base']
        system_prompt = config['system_prompt']
        model = Client(system_prompt, model_name, model_api_key, model_api_base)

        while True:
            query = input("输入（输入“退出”可退出对话）：")
            if query == "退出":
                break
            processor = SearchResultProcessor(api_key, query, search_params)
            processed_documents = processor.process(serper_api_url)
            all_page_content = " ".join([doc.page_content for doc in processed_documents])
            cleaned_all_page_content = processor.clean_text(all_page_content)
            response = model(str(cleaned_all_page_content) + query, 'ChainOfThought')
            print(response)
    