import folder_paths
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import os
import json
import io
from comfy.cli_args import args

class SaveImageNotPreview(object):
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_full": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "SelfNodes"

    def save_images(self, images, filename_full="", filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            if filename_full != "" and not os.path.exists(os.path.join(full_output_folder, f"{filename_full}.png")):
                filename_with_batch_num = filename_full
            else:
                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                filename_with_batch_num = f"{filename_full}{filename_with_batch_num}_{counter:05}_"
            file = f"{filename_with_batch_num}.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return ()

class SaveImageJPG(object):
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "SelfNodes"

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()

        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Create a dictionary to store metadata
            exif_data = {}
            if prompt is not None:
                exif_data["prompt"] = prompt
            # if extra_pnginfo is not None:
            #     for key, value in extra_pnginfo.items():
            #         exif_data[key] = value

            # Convert EXIF data dictionary to a format suitable for saving
            exif_bytes = json.dumps(exif_data).encode('utf-8')
            # print(exif_bytes)

            # Save the image with EXIF data
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            filename_with_batch_num = f"{filename_with_batch_num}_{counter:05}_"
            file = f"{filename_with_batch_num}.jpg"
            img.save(os.path.join(full_output_folder, file), format="JPEG", exif=exif_bytes)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"result": (filename_with_batch_num, ), "ui": { "images": results } }

class SaveImageWEBP(object):
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_full": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }


    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "SelfNodes"

    def save_images(self, images, filename_full="", filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()

        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Save the image with EXIF data
            if filename_full != "":
                if filename_full.endswith(".webp"):
                    filename_with_batch_num = filename_full
                    file = filename_full
                else:
                    if not os.path.exists(os.path.join(full_output_folder, f"{filename_full}.webp")):
                        filename_with_batch_num = filename_full
                    else:
                        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                        filename_with_batch_num = f"{filename_full}{filename_with_batch_num}_{counter:05}_"
                    file = f"{filename_with_batch_num}.webp"
            else:
                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                filename_with_batch_num = f"{filename_full}{filename_with_batch_num}_{counter:05}_"
                file = f"{filename_with_batch_num}.webp"

            img.save(os.path.join(full_output_folder, file), format="webp")
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"result": (filename_with_batch_num, ), "ui": { "images": results } }

NODE_CLASS_MAPPINGS = {
    "保存图片JPG": SaveImageJPG,
    "保存图片WEBP": SaveImageWEBP,
    "保存图片不预览": SaveImageNotPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "保存图片JPG": "保存图片JPG",
    "保存图片WEBP": "保存图片WEBP",
    "保存图片不预览": "保存图片不预览",
}