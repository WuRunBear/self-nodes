import requests
import hashlib
import random
import json
import time

class DisruptText:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
              "string": ("STRING", {
                  "multiline": True,
                  "default": ""
              }),
              "delimiter": ("STRING", {
                  "multiline": False,
                  "default": ","
              }),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "disrupt_text"

    CATEGORY = "SelfNodes"

    def disrupt_text(self, string, delimiter):
      # 使用指定的分隔符分割字符串
      parts = string.split(delimiter)
      
      # 打乱分割后的子字符串顺序
      random.shuffle(parts)
      
      # 组合打乱后的子字符串为新的字符串
      shuffled_string = delimiter.join(parts)

      shuffled_string = shuffled_string+','

      return (shuffled_string,)

NODE_CLASS_MAPPINGS = {
    "打乱文本": DisruptText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "打乱文本": "打乱文本"
}