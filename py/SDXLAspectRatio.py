import torch

try:
    import comfy.model_management as model_management
except Exception:
    model_management = None


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

        device = model_management.get_torch_device() if model_management is not None else "cpu"
        latent = torch.zeros((batch_size, 4, height // 8, width // 8), device=device)

        show_help = f"aspect_ratio={aspect_ratio}, swap_dimensions={swap_dimensions}, latent={batch_size}x4x{height//8}x{width//8}"
        return (width, height, float(upscale_factor), int(batch_size), {"samples": latent}, show_help)


NODE_CLASS_MAPPINGS = {
    "SelfNodes_SDXLAspectRatio": SelfNodes_SDXLAspectRatio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfNodes_SDXLAspectRatio": "SDXL比例/空Latent",
}

