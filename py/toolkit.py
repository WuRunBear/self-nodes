import comfy.model_management
import requests
import hashlib
import random
import json
import time
import os
import folder_paths
try:
    import torch
except Exception:
    torch = None

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


class SelfNodes_SDXLAspectRatio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = [
            "custom",
            "1:1 square 1024x1024",
            "3:4 portrait 896x1152",
            "5:8 portrait 832x1216",
            "9:16 portrait 768x1344",
            "9:21 portrait 640x1536",
            "4:3 landscape 1152x896",
            "3:2 landscape 1216x832",
            "16:9 landscape 1344x768",
            "21:9 landscape 1536x640",
        ]

        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "aspect_ratio": (aspect_ratios,),
                "swap_dimensions": (["Off", "On"],),
                "upscale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "INT", "LATENT", "STRING")
    RETURN_NAMES = ("width", "height", "upscale_factor", "batch_size", "empty_latent", "show_help")
    FUNCTION = "aspect_ratio"
    CATEGORY = "SelfNodes/工具"

    def aspect_ratio(self, width, height, aspect_ratio, swap_dimensions, upscale_factor, batch_size):
        if torch is None:
            raise RuntimeError("torch is required")

        presets = {
            "1:1 square 1024x1024": (1024, 1024),
            "3:4 portrait 896x1152": (896, 1152),
            "5:8 portrait 832x1216": (832, 1216),
            "9:16 portrait 768x1344": (768, 1344),
            "9:21 portrait 640x1536": (640, 1536),
            "4:3 landscape 1152x896": (1152, 896),
            "3:2 landscape 1216x832": (1216, 832),
            "16:9 landscape 1344x768": (1344, 768),
            "21:9 landscape 1536x640": (1536, 640),
        }

        if aspect_ratio in presets:
            width, height = presets[aspect_ratio]

        if swap_dimensions == "On":
            width, height = height, width

        width = max(64, int(width))
        height = max(64, int(height))
        width = (width // 8) * 8
        height = (height // 8) * 8

        device = comfy.model_management.get_torch_device()
        latent = torch.zeros((batch_size, 4, height // 8, width // 8), device=device)

        show_help = f"aspect_ratio={aspect_ratio}, swap_dimensions={swap_dimensions}, latent={batch_size}x4x{height//8}x{width//8}"
        return (width, height, float(upscale_factor), int(batch_size), {"samples": latent}, show_help)


#---------------------------------------------------------------------------------------------------------------------#
# MAPPINGS
#---------------------------------------------------------------------------------------------------------------------#
# For reference only, actual mappings are in __init__.py

NODE_CLASS_MAPPINGS = {
    "停止当前队列": SelfNodes_StopCurrentQueue,
    "范围随机整数": randomToFixedLength,
    "限制比例大小": LimitRatioSize,
    "加载lora（可从路径加载）": SelfNodes_LoraLoader,
    "宽高比例": SelfNodes_SDXLAspectRatio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "停止当前队列": "停止当前队列",
    "范围随机整数": "范围随机整数",
    "限制比例大小": "限制比例大小",
    "加载lora（可从路径加载）": "加载lora（可从路径加载）",
    "宽高比例": "宽高比例"
}
