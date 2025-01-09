import comfy.model_management
import requests
import hashlib
import random
import json
import time

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any_type = AnyType("*")

class SelfNodes_StopCurrentQueue(object):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (any_type, {}),
                "boolean": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (any_type,)

    FUNCTION = "stop"

    CATEGORY = "SelfNodes/工具"

    def stop(self, any, boolean):
        if boolean:
          comfy.model_management.interrupt_current_processing(True)
        return (any,)


class randomToFixedLength:
    def __init__(self):
        # self.randomN=None
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
              "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}) ,
              "min_length": ("INT", {"default": 1, "min": 0, "max": 9999, "step": 1}),
              "max_length": ("INT", {"default": 1, "min": 0, "max": 9999, "step": 1}),
              "multiple": ("INT", {"default": 32, "min": 1, "max": 9999, "step": 1}),
              # "control_method": (["random", "fixed"],),
            },
        }

    RETURN_TYPES = ("INT",)

    FUNCTION = "random_to_fixed_length"

    CATEGORY = "SelfNodes/工具"

    def random_to_fixed_length(self, seed, min_length, max_length, multiple, ):
        # 找到范围内的最小倍数
        min_multiple = (min_length + multiple - 1) // multiple * multiple
        if min_multiple > max_length:
            raise ValueError("指定的范围内没有符合条件的倍数")
        # 在范围内随机选择一个倍数

        random.seed(seed)
        # self.randomN = random.randrange(min_multiple, max_length + 1, multiple)
        return (random.randrange(min_multiple, max_length + 1, multiple), )

    # @classmethod
    # def IS_CHANGED(self, min_length, max_length, multiple, control_method):
    #     if control_method == "random":
    #         return time.time()
    #     elif control_method == "fixed":
    #         return self.randomN

# 限制比例大小
class LimitRatioSize:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
              "widht": ("INT", {"default": 1, "min": 0, "max": 9999, "step": 1}),
              "height": ("INT", {"default": 1, "min": 0, "max": 9999, "step": 1}),
              "min_length": ("INT", {"default": 1024, "min": 1, "max": 9999, "step": 1}),
              "max_length": ("INT", {"default": 1024, "min": 1, "max": 9999, "step": 1}),
              "multiple": ("INT", {"default": 32, "min": 1, "max": 9999, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NMAES = ("widht", "height",)

    FUNCTION = "limit_ratio_size"

    CATEGORY = "SelfNodes/工具"

    def limit_ratio_size(self, widht, height, min_length, max_length, multiple):
        # 计算比例因子以确保宽高满足 min_length
        if max(widht, height) < min_length:
            scale_factor = min_length / max(widht, height)
            widht_scaled = int(widht * scale_factor)
            height_scaled = int(height * scale_factor)
        else:
            widht_scaled, height_scaled = widht, height

        # 确保宽高是 multiple 的倍数
        widht_scaled = (widht_scaled // multiple) * multiple
        height_scaled = (height_scaled // multiple) * multiple

        # 如果计算后的尺寸大于 max_length，进行缩小
        if max(widht_scaled, height_scaled) > max_length:
            scale_factor = max_length / max(widht_scaled, height_scaled)
            widht_scaled = int(widht_scaled * scale_factor)
            height_scaled = int(height_scaled * scale_factor)

            # 再次调整为 multiple 的倍数
            widht_scaled = (widht_scaled // multiple) * multiple
            height_scaled = (height_scaled // multiple) * multiple

        return (widht_scaled, height_scaled,)


#---------------------------------------------------------------------------------------------------------------------#
# MAPPINGS
#---------------------------------------------------------------------------------------------------------------------#
# For reference only, actual mappings are in __init__.py

NODE_CLASS_MAPPINGS = {
    "停止当前队列": SelfNodes_StopCurrentQueue,
    "范围随机整数": randomToFixedLength,
    "限制比例大小": LimitRatioSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "停止当前队列": "停止当前队列",
    "范围随机整数": "范围随机整数",
    "限制比例大小": "限制比例大小"
}