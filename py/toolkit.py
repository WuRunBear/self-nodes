import comfy.model_management
import requests
import hashlib
import random
import json
import time
import os
import folder_paths

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

class SelfNodes_LoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "SelfNodes/工具"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = lora_name
        lora = None
        if not os.path.isabs(lora_name):
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_path)
            if self.loaded_lora is not None:
                if self.loaded_lora[0] == lora_path:
                    lora = self.loaded_lora[1]
                else:
                    self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)


#---------------------------------------------------------------------------------------------------------------------#
# MAPPINGS
#---------------------------------------------------------------------------------------------------------------------#
# For reference only, actual mappings are in __init__.py

NODE_CLASS_MAPPINGS = {
    "停止当前队列": SelfNodes_StopCurrentQueue,
    "范围随机整数": randomToFixedLength,
    "限制比例大小": LimitRatioSize,
    "加载lora（可从路径加载）": SelfNodes_LoraLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "停止当前队列": "停止当前队列",
    "范围随机整数": "范围随机整数",
    "限制比例大小": "限制比例大小",
    "加载lora（可从路径加载）": "加载lora（可从路径加载）"
}