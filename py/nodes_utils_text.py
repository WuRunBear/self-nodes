
import os
import csv
import io
import re
import requests
import hashlib
import random
import json
import time

class AnyType(str):
    """A special type that can be connected to any other types. SelfNodesedit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

def is_child_dir(parent_path, child_path):
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)
    return os.path.commonpath([parent_path]) == os.path.commonpath([parent_path, child_path])

def get_file(root_dir, file):
    if file == "[none]" or not file or not file.strip():
        raise ValueError("No file")

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    full_path = os.path.join(root_dir, file)

    if not is_child_dir(root_dir, full_path):
        raise ReferenceError()

    return full_path

#---------------------------------------------------------------------------------------------------------------------#
# Text Util Nodes
#---------------------------------------------------------------------------------------------------------------------#
class SelfNodes_SplitString:

    @classmethod
    def INPUT_TYPES(s):  
    
        return {"required": {
                    "text": ("STRING", {"multiline": False, "default": "text"}),
                },
                "optional": {
                    "delimiter": ("STRING", {"multiline": False, "default": ","}),
                }            
        }

    RETURN_TYPES = (any_type, any_type, any_type, any_type, "STRING", )
    RETURN_NAMES = ("string_1", "string_2", "string_3", "string_4", )    
    FUNCTION = "split"
    CATEGORY = "SelfNodes/文本"

    def split(self, text, delimiter=""):

        # Split the text string
        parts = text.split(delimiter)
        strings = [part.strip() for part in parts[:4]]
        string_1, string_2, string_3, string_4 = strings + [""] * (4 - len(strings))            

        return (string_1, string_2, string_3, string_4,  )

#---------------------------------------------------------------------------------------------------------------------#
class SelfNodes_Text:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": True}),
            }
        }

    RETURN_TYPES = (any_type, "STRING", )
    RETURN_NAMES = ("text", )
    FUNCTION = "text_multiline"
    CATEGORY = "SelfNodes/文本"

    def text_multiline(self, text):
            
        return (text, )

#---------------------------------------------------------------------------------------------------------------------#
class SelfNodes_MultilineText:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": True}),
                "convert_from_csv": ("BOOLEAN", {"default": False}),
                "csv_quote_char": ("STRING", {"default": "'", "choices": ["'", '"']}),
                "remove_chars": ("BOOLEAN", {"default": False}),
                "chars_to_remove": ("STRING", {"multiline": False, "default": ""}),
                "split_string": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (any_type, "STRING", )
    RETURN_NAMES = ("multiline_text", )
    FUNCTION = "text_multiline"
    CATEGORY = "SelfNodes/文本"

    def text_multiline(self, text, chars_to_remove, split_string=False, remove_chars=False, convert_from_csv=False, csv_quote_char="'"):
    
        new_text = []

        # Remove trailing commas
        text = text.rstrip(',')

        if convert_from_csv:
            # Convert CSV to multiline text
            csv_reader = csv.reader(io.StringIO(text), quotechar=csv_quote_char)
            for row in csv_reader:
                new_text.extend(row)       
        if split_string: 
            if text.startswith("'") and text.endswith("'"):
                text = text[1:-1]  # Remove outer single quotes
                values = [value.strip() for value in text.split("', '")]
                new_text.extend(values)
            elif text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]  # Remove outer single quotes
                    values = [value.strip() for value in text.split('", "')]
                    new_text.extend(values)   
            elif ',' in text and text.count("'") % 2 == 0:
                # Assume it's a list-like string and split accordingly
                text = text.replace("'", '')  # Remove single quotes
                values = [value.strip() for value in text.split(",")]
                new_text.extend(values)
            elif ',' in text and text.count('"') % 2 == 0:
                    # Assume it's a list-like string and split accordingly
                    text = text.replace('"', '')  # Remove single quotes
                    values = [value.strip() for value in text.split(",")]
                    new_text.extend(values)                 
        if convert_from_csv == False and split_string == False:
            # Process multiline text
            for line in io.StringIO(text):    
                if not line.strip().startswith('#'):
                    if not line.strip().startswith("\n"):
                        line = line.replace("\n", '')
                    if remove_chars:
                        # Remove quotes from each line
                        line = line.replace(chars_to_remove, '')
                    new_text.append(line)                

        new_text = "\n".join(new_text)
        
        return (new_text, )

#---------------------------------------------------------------------------------------------------------------------# 
class SelfNodes_SaveTextToFile:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "root_dir": ("STRING", {"default": ""}),
                        "file": ("STRING", {"default": "file.txt"}),
                        "append": (["append", "overwrite", "new only"], {}),
                        "insert": ("BOOLEAN", {
                            "default": True, "label_on": "new line", "label_off": "none",
                        }),
                        "text": ("STRING", {"forceInput": True, "multiline": True})
                    }
        }
        
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = () 
    FUNCTION = 'save_list'
    CATEGORY = "SelfNodes/文本"
    OUTPUT_NODE = True

    def save_list(self, **kwargs):
        self.file = get_file(kwargs["root_dir"], kwargs["file"])
        if kwargs["append"] == "new only" and os.path.exists(self.file):
            raise FileExistsError(
                self.file + " already exists and 'new only' is selected.")
        with open(self.file, "a+" if kwargs["append"] == "append" else "w") as f:
            is_append = f.tell() != 0
            if is_append and kwargs["insert"]:
                f.write("\n")
            f.write(kwargs["text"])

        with open(self.file, "r") as f:
            return (f.read(), )

#---------------------------------------------------------------------------------------------------------------------#
class SelfNodes_TextConcatenate:

    @ classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                },
                "optional": {
                "text1": ("STRING", {"multiline": False, "default": "", "forceInput": True}),                
                "text2": ("STRING", {"multiline": False, "default": "", "forceInput": True}), 
                "separator": ("STRING", {"multiline": False, "default": ""}),                
            },
        }

    RETURN_TYPES = (any_type, "STRING", )
    RETURN_NAMES = ("STRING",)
    FUNCTION = "concat_text"
    CATEGORY = "SelfNodes/文本"

    def concat_text(self, text1="", text2="", separator=""):
    
        return (text1 + separator + text2, )

#---------------------------------------------------------------------------------------------------------------------#
class SelfNodes_TextReplace:

    @ classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),            
                },
            "optional": {
                "find1": ("STRING", {"multiline": False, "default": ""}),
                "replace1": ("STRING", {"multiline": False, "default": ""}),
                "find2": ("STRING", {"multiline": False, "default": ""}),
                "replace2": ("STRING", {"multiline": False, "default": ""}),
                "find3": ("STRING", {"multiline": False, "default": ""}),
                "replace3": ("STRING", {"multiline": False, "default": ""}),    
            },
        }

    RETURN_TYPES = (any_type, "STRING", )
    RETURN_NAMES = ("STRING", )
    FUNCTION = "replace_text"
    CATEGORY = "SelfNodes/文本"

    def replace_text(self, text, find1="", replace1="", find2="", replace2="", find3="", replace3=""):
    
        text = text.replace(find1, replace1)
        text = text.replace(find2, replace2)
        text = text.replace(find3, replace3)
        
        return (text,)    

#---------------------------------------------------------------------------------------------------------------------#
class SelfNodes_TextBlacklist:

    @ classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "blacklist_words": ("STRING", {"multiline": True, "default": ""}),
                },
            "optional": {
                "replacement_text": ("STRING", {"multiline": False, "default": ""}),    
            },
        }

    RETURN_TYPES = (any_type, "STRING", )
    RETURN_NAMES = ("STRING", )
    FUNCTION = "replace_text"
    CATEGORY = "SelfNodes/文本"

    def replace_text(self, text, blacklist_words, replacement_text=""):
        text_out = text 

        for line in blacklist_words.split('\n'):  # Splitting based on line return
            if line.strip():
                while re.search(line.strip(), text_out):
                    text_out = re.sub(line.strip(), replacement_text, text_out)
        text_out += ","

        return (text_out, )   

#---------------------------------------------------------------------------------------------------------------------#
class SelfNodes_TextOperation:

    @ classmethod
    def INPUT_TYPES(cls):
      
        operations = ["uppercase", "lowercase", "capitalize", "invert_case", "reverse", "trim", "remove_spaces"]
    
        return {
            "required": {
                "text": ("STRING", {"multiline": False, "default": "", "forceInput": True}),            
                "operation": (operations,),
            },
        }

    RETURN_TYPES = (any_type, "STRING", )
    RETURN_NAMES = ("STRING", )
    FUNCTION = "text_operation"
    CATEGORY = "SelfNodes/文本"

    def text_operation(self, text, operation):
    
        if operation == "uppercase":
            text_out = text.upper()
        elif operation == "lowercase":
            text_out = text.lower()
        elif operation == "capitalize":
            text_out = text.capitalize()
        elif operation == "invert_case":
            text_out = text.swapcase()
        elif operation == "reverse":
            text_out = text[::-1]
        elif operation == "trim":
            text_out = text.strip()
        elif operation == "remove_spaces":
            text_out = text.replace(" ", "")
        else:
            return "SelfNodes Text Operation: Invalid operation."

        return (text_out, )

#---------------------------------------------------------------------------------------------------------------------#
class SelfNodes_TextLength:

    @ classmethod
    def INPUT_TYPES(cls):
         
        return {
            "required": {
                "text": ("STRING", {"multiline": False, "default": "", "forceInput": True}),            
            },
        }

    RETURN_TYPES = ("INT", "STRING", )
    RETURN_NAMES = ("INT", )
    FUNCTION = "len_text"
    CATEGORY = "SelfNodes/文本"

    def len_text(self, text):
    
        int_out = len(text)

        return (int_out, )
#---------------------------------------------------------------------------------------------------------------------#
class SelfNodes_SplitTagsInsertText:

    @ classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "text": ("STRING", {"multiline": False, "default": ""}),
                "delimiter": ("STRING", {"multiline": False, "default": ","}),
                "index": ("INT", {"default": 0, "step": 1}),
                "insert_text": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("STRING", )
    FUNCTION = "fun_text"
    CATEGORY = "SelfNodes/文本"

    def fun_text(self, text, delimiter=",", index=0, insert_text=""):
        def insert_char_at_position(original_string, char_to_insert, position):
            # 处理负索引
            if position < 0:
                position += len(original_string) + 1

            # 如果字符串开头是(或[，则插入到(或[后面
            if position == 0 and (original_string[0] == '(' or original_string[0] == '['):
                position += 1
            # 检查位置是否在字符串长度范围内
            if position < 0 or position > len(original_string):
                raise ValueError("位置超出了字符串范围")

            # 使用切片将字符插入到指定位置
            new_string = original_string[:position] + char_to_insert + original_string[position:]
            return new_string

        out = ""
        parts = text.split(delimiter)
        # 循环parts，找到index位置的元素，插入insert_text
        for i, part in enumerate(parts):
            if part.strip() == "":
                continue
            parts[i] = insert_char_at_position(part.strip(), insert_text, index)

        out = delimiter.join(parts)

        return (out, )
#---------------------------------------------------------------------------------------------------------------------#
class SelfNodes_SplitTagsRemoveDuplication:

    @ classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "text": ("STRING", {"multiline": False, "default": ""}),
                "delimiter": ("STRING", {"multiline": False, "default": ","}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("STRING", )
    FUNCTION = "fun_text"
    CATEGORY = "SelfNodes/文本"

    def fun_text(self, text, delimiter=","):
        split_text = [item.strip() for item in text.split(delimiter)]  # 分割并去除每项的前后空白

        unique_text = []
        seen = set()
        for item in split_text:
            if item not in seen:
                seen.add(item)
                unique_text.append(item)

        result = delimiter.join(unique_text)  # 转回字符串
        return (result, )
#---------------------------------------------------------------------------------------------------------------------#
class SelfNodes_PatternSearchReturn:

    @ classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "pattern": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("STRING", )
    FUNCTION = "fun_text"
    CATEGORY = "SelfNodes/文本"

    def fun_text(self, text, pattern=""):
        match = re.search(pattern, text)
        if match:
            return (match.group(1), )
        return ("", )

class SelfNodes_randomReturnTags(object):
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}) ,
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "delimiter": ("STRING", {
                    "multiline": False,
                    "default": ","
                }),
                "num_items": ("INT", {"default": 3, "min": 0, "max": 9999, "step": 1}),
                "random_weight": ("BOOLEAN", {"default": True}),
                "min_weight": ("FLOAT", {"default": 0.3, "step": 0.05}),
                "max_weight": ("FLOAT", {"default": 1.3, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)

    FUNCTION = "split_and_randomize"

    CATEGORY = "SelfNodes/文本"

    def split_and_randomize(self, seed, input_text, delimiter, num_items, random_weight, min_weight, max_weight,):
        items = input_text.split(delimiter)

        random.seed(seed)

        if num_items is None:
            selected_items = random.sample(items, 3)
        else:
            if num_items > len(items):
                selected_items = items
            else:
                selected_items = random.sample(items, num_items)

        weight = round(random.uniform(min_weight, max_weight), 2)
        for i in range(len(selected_items)):
            if random_weight:
                if round(random.uniform(1,num_items)) != round(num_items/2):
                    weight = round(random.uniform(min_weight, max_weight), 2)  # 使用 min_weight 和 max_weight 限制权重范围
                selected_items[i] = f"({selected_items[i]}:{weight})"

        result_string = ','.join(selected_items)

        return (result_string,)

    # @classmethod
    # def IS_CHANGED(self, input_text, num_items,):
    #     if control_method == "random":
    #         return time.time()
    #     elif control_method == "fixed":
    #         return self.randomN

class SelfNodes_DisruptText:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
              "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}) ,
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

    CATEGORY = "SelfNodes/文本"

    def disrupt_text(self, seed, string, delimiter):
      # 使用指定的分隔符分割字符串
      parts = string.split(delimiter)
      
      random.seed(seed)

      # 打乱分割后的子字符串顺序
      random.shuffle(parts)

      # 组合打乱后的子字符串为新的字符串
      shuffled_string = delimiter.join(parts)

      shuffled_string = shuffled_string+','

      return (shuffled_string,)


class SelfNodes_BaiduTranslate:
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
                "app_id": ("STRING", {
                    "default": ""
                }),
                "app_key": ("STRING", {
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "translate"

    CATEGORY = "SelfNodes/文本"

    def translate(self, input_text, from_lang, to_lang, app_id, app_key):
        if not input_text:
            return ("",)

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


class SelfNodes_StringToDTGParams(object):
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

    CATEGORY = "SelfNodes/文本"

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


class SelfNodes_LoadTextList(object):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}) ,
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "start_index": ("INT", {"default": -1, "min": -1, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING")

    FUNCTION = "loadTextList"

    CATEGORY = "SelfNodes/文本"

    def loadTextList(self, seed, directory: str, start_index: int = 0):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory} cannot be found.'")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        random.seed(seed)

        # Filter files by extension
        valid_extensions = ['.txt']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        # start at start_index
        if start_index>=0 and dir_files[start_index]:
            dir_files = dir_files[start_index]
        else:
            dir_files = dir_files[random.randint(0, len(dir_files)-1)]

        # 加载文件内容
        result_dict = ""
        with open(dir_files, 'r', encoding='utf-8') as f:
            result_dict = f.read()
        return (result_dict,)


#---------------------------------------------------------------------------------------------------------------------#
# MAPPINGS
#---------------------------------------------------------------------------------------------------------------------#
# For reference only, actual mappings are in __init__.py

NODE_CLASS_MAPPINGS = {
    ### Utils Text
    "SelfNodes Text": SelfNodes_Text,
    "SelfNodes Multiline Text": SelfNodes_MultilineText,
    "SelfNodes Split String": SelfNodes_SplitString,
    "SelfNodes Text Concatenate": SelfNodes_TextConcatenate,
    "SelfNodes Text Replace": SelfNodes_TextReplace,
    "SelfNodes Text Blacklist": SelfNodes_TextBlacklist,
    "SelfNodes Text Length": SelfNodes_TextLength,
    "SelfNodes Text Operation": SelfNodes_TextOperation,
    "SelfNodes Save Text To File": SelfNodes_SaveTextToFile,
    "SelfNodes Split Tags Insert Text": SelfNodes_SplitTagsInsertText,
    "SelfNodes Split Tags Remove Duplication": SelfNodes_SplitTagsRemoveDuplication,
    "SelfNodes Pattern Search Return": SelfNodes_PatternSearchReturn,
    "随机返回tags": SelfNodes_randomReturnTags,
    "打乱文本": SelfNodes_DisruptText,
    "百度翻译": SelfNodes_BaiduTranslate,
    "转成DTG参数": SelfNodes_StringToDTGParams,
    "文件夹加载txt文件": SelfNodes_LoadTextList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    ### Utils Text
    "SelfNodes Text": "文本",
    "SelfNodes Multiline Text": "多行文本",
    "SelfNodes Split String": "拆分字符串",
    "SelfNodes Text Concatenate": "文本连接",
    "SelfNodes Text Replace": "文本替换",
    "SelfNodes Text Blacklist": "文本黑名单",
    "SelfNodes Text Length": "文本长度",
    "SelfNodes Text Operation": "文本操作",
    "SelfNodes Save Text To File": "将文本保存到文件",
    "SelfNodes Split Tags Insert Text": "分割标签并插入文本",
    "SelfNodes Split Tags Remove Duplication": "分割标签并去重",
    "SelfNodes Pattern Search Return": "正则匹配内容输出",
    "随机返回tags": "随机返回tags",
    "打乱文本": "打乱文本",
    "百度翻译": "百度翻译",
    "转成DTG参数": "转成DTG参数",
    "文件夹加载txt文件": "文件夹加载txt文件"
}