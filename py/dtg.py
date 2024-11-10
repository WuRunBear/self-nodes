import requests
import hashlib
import random
import json
import time
import re

class splitStringToDTGParams(object):
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": "<艺术家列表>: \n<人物特征>: \n<特殊标签>: \n<生成>: "
                }),
            },
        }

    RETURN_TYPES = ("STRING","STRING","STRING","STRING",)
    RETURN_NAMES = ("artist", "characters", "special_tags", "general")

    FUNCTION = "splitString"

    CATEGORY = "SelfNodes"

    def splitString(self, input_text):
        # Generate a random integer of arbitrary length
        matches = re.split(r'(<[^>]*>):', input_text)
        print(matches)
        # 初始化一个空字典
        result_dict = {}

# \((?:[^()\\]*|\\.)*:\d+(\.\d+)?\)
        # 遍历列表并按模式分组
        i = 0
        while i < len(matches):
            if re.match(r'<.*?>', matches[i]):  # 如果当前项是尖括号内容
                key = matches[i][1:-1]  # 去掉尖括号
                value_parts = []
                i += 1
                while i < len(matches) and not re.match(r'<.*?>', matches[i]):
                    value_parts.append(matches[i])
                    i += 1
                result_dict[key] = ' '.join(value_parts)  # 将值合并成一个字符串
            else:
                i += 1

        return (result_dict["艺术家列表"], result_dict["人物特征"], result_dict["特殊标签"], result_dict["生成"],)

NODE_CLASS_MAPPINGS = {
    "转成DTG参数": splitStringToDTGParams
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "转成DTG参数": "转成DTG参数"
}