import requests
import hashlib
import random
import json
import time
import re

class randomReturnTags(object):
    randomN=None
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
                "num_items": ("INT", {"default": 3, "min": 0, "max": 9999, "step": 1}),
                "control_method": (["random", "fixed"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("artist",)

    FUNCTION = "split_and_randomize"

    CATEGORY = "SelfNodes"

    def split_and_randomize(self, input_text, num_items, control_method):
        items = input_text.split(',')

        if num_items is None:
            selected_items = random.sample(items, 3)
        else:
            if num_items > len(items):
                selected_items = items
            else:
                selected_items = random.sample(items, num_items)

        result_string = ','.join(selected_items)

        self.randomN = result_string
        return (result_string,)

    @classmethod
    def IS_CHANGED(self, input_text, num_items, control_method):
        if control_method == "random":
            return time.time()
        elif control_method == "fixed":
            return self.randomN

NODE_CLASS_MAPPINGS = {
    "随机返回tags": randomReturnTags
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "随机返回tags": "随机返回tags"
}