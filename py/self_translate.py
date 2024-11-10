import requests
import hashlib
import random
import json
import time

class SelfTranslate:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "from_lang": ("STRING", {
                    "default": "auto"
                }),
                "to_lang": ("STRING", {
                    "default": "en"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "translate"

    CATEGORY = "SelfNodes"

    def translate(self, input_text, from_lang, to_lang):
        if not input_text:
            return ("",)

        app_id = "20230424001654006"
        app_key = "gICCYNdWlueE9KWDNVSv"

        # 需要等待一秒,不然多个节点可能会出现频繁请求被拒绝的情况
        time.sleep(1)

        api_url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
        
        salt = str(random.randint(32768, 65536))
        sign = hashlib.md5((app_id + input_text + salt + app_key).encode()).hexdigest()
        
        params = {
            "q": input_text,
            "from": from_lang,
            "to": to_lang,
            "appid": app_id,
            "salt": salt,
            "sign": sign
        }
        print("翻译参数：")
        print({
            "q": input_text,
            "from": from_lang,
            "to": to_lang,
            "appid": app_id,
            "salt": salt,
            "sign": sign
        })
        response = requests.get(api_url, params=params)
        print(response.text)
        response_json = json.loads(response.text)
        translation_list = [result["dst"] for result in response_json["trans_result"]]
        translation = "\n".join(translation_list).lower()
        
        print("翻译结果：")
        print(translation)
        return (translation,)

NODE_CLASS_MAPPINGS = {
    "自用文本翻译": SelfTranslate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "自用文本翻译": "自用文本翻译"
}