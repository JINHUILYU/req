import os
import re
from time import sleep
from flask import jsonify
from openai import OpenAI


class LLM:
    def __init__(self, model='gpt35', temperature=0.3, base_url='https://api.chatanywhere.tech/v1',
                 key=''):
        self.history = []
        self.client = OpenAI(
            api_key=key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.models = {
            "kimi": "moonshot-v1-8k",
            "gpt35": "gpt-3.5-turbo",
            "gpt4o": "gpt-4o",
            "gpt4": "gpt-4"
        }
        self.base_urls = {
            "kimi": "https://api.moonshot.cn/v1",
            # "gpt35": "https://api.openai.com/v1",  # 官方接口
            "gpt35": "https://api.chatanywhere.tech/v1",  # 转发接口
            # "gpt4": "https://api.openai.com/v1",  # 官方接口
            "gpt4": "https://api.chatanywhere.tech/v1",  # 转发接口
            # "gpt4o": "https://api.openai.com/v1",  # 官方接口
            "gpt4o": "https://api.chatanywhere.tech/v1",  # 转发接口
        }
        self.prompt = ''''''
        self.history.append({"role": "system", "content": self.prompt})

    def query(self, sentence, context=True):
        self.history.append({"role": "user", "content": sentence})
        try:
            completion = self.client.chat.completions.create(
                model=self.models.get(self.model),
                messages=self.history,
                temperature=self.temperature,
                timeout=15,
            )
        except Exception as ex:
            print(f"Error processing data: {ex}")
            return jsonify({"error": str(ex)})
        # 获取处理后的数据
        processed_data = completion.choices[0].message.content.strip()
        # 上下文处理
        if context:
            self.history.append({"role": "assistant", "content": processed_data})
        else:
            self.history.pop()
        return processed_data
        # 与前端交互时返回json数据
        # return jsonify({"result": processed_data})
