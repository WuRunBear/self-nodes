
import os
import csv
import io
import re

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
    CATEGORY = "SelfNodes"

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
    CATEGORY = "SelfNodes"

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
    CATEGORY = "SelfNodes"

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
    CATEGORY = "SelfNodes"
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
    CATEGORY = "SelfNodes"

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
    CATEGORY = "SelfNodes"

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
    CATEGORY = "SelfNodes"

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
    CATEGORY = "SelfNodes"

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
    CATEGORY = "SelfNodes"

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
    CATEGORY = "SelfNodes"

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
    CATEGORY = "SelfNodes"

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
}