import folder_paths
from PIL import Image, ImageOps, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import os
import json
from io import BytesIO
from comfy.cli_args import args
import torch
import torch.nn as nn
from torchvision import transforms
import pytorch_lightning as pl
import clip
import requests
import random
from imgutils.detect import detect_person

class AnyType(str):
    """A special type that can be connected to any other types. SelfNodesedit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0) 
    
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

    CATEGORY = "SelfNodes/图片"

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

            _, ext = os.path.splitext(filename_full)
            if not bool(ext):
                filename_full = f"{filename_full}.png"

            if filename_full != "" and not os.path.exists(os.path.join(full_output_folder, f"{filename_full}")):
                filename_with_batch_num = filename_full
            else:
                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                filename_with_batch_num = f"{_}{filename_with_batch_num}_{counter:05}_"
            file = f"{filename_with_batch_num}"
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

    CATEGORY = "SelfNodes/图片"

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

    CATEGORY = "SelfNodes/图片"

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

class TextImg(object):

    @classmethod
    def INPUT_TYPES(cls):
        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith(".ttf")]

        return {
            "required": {
                # "width": ("INT", {"default": 1024, "min": 64, "max": 2048}),
                # "height": ("INT", {"default": 1024, "min": 64, "max": 2048}),
                "images": ("IMAGE", ),
                "text": ("STRING", {"multiline": False, "default": ""}),
                "position": (["top", "bottom", "left", "right"],  {"default": "top"}),
                "font_name": (file_list,),
            },
        }


    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "SelfNodes/图片"

    def save_images(self, images, text, position, font_name,):
        def create_text_image(text, font_path, font_size, width, height, position='bottom'):
            # 创建一个字体对象
            font = ImageFont.truetype(font_path, font_size)

            # 使用 ImageDraw 计算文本的边界框
            dummy_image = Image.new('RGB', (1, 1))  # 创建一个最小的图像用于计算文本大小
            draw = ImageDraw.Draw(dummy_image)
            bbox = draw.textbbox((0, 0), text, font=font)  # 获取文本的边界框
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # text_width = width
            # text_height = height
            if position=="top" or position=="bottom":
                text_width = width
                text_height = 100
        
            if position=="left" or position=="right":
                text_width = 100
                text_height = height
        
            print(f"{width}： {height}")
            print(f"{text_width}： {text_height}")
            # 创建合适大小的图像并绘制文本
            text_image = Image.new('RGB', (text_width, text_height), (255, 255, 255))  # 创建一个背景为白色的图片
            draw = ImageDraw.Draw(text_image)

            # 在图片上居中绘制文本
            text_position = ((text_width - text_width) // 2, (text_height - text_height) // 2)
            draw.text(text_position, text, font=font, fill=(0, 0, 0))  # 绘制文本，颜色为黑色

            return text_image
        
        def concatenate_images(base_image, text_image, position='bottom'):
            # if position == 'right':
            #   new_image = torch.cat((text_image, base_image), dim=2)
            # elif position == 'bottom':
            #   new_image = torch.cat((text_image, base_image), dim=1)
            # elif position == 'left':
            #   new_image = torch.cat((base_image, text_image), dim=2)
            # elif position == 'top':
            #   new_image = torch.cat((base_image, text_image), dim=1)

            # 获取拼接后图像的尺寸
            base_width, base_height = base_image.size
            text_width, text_height = text_image.size
            
            if position == 'bottom':
                # 拼接在底部
                new_image = Image.new('RGB', (base_width, base_height + text_height), (255, 255, 255))
                new_image.paste(base_image, (0, 0))
                new_image.paste(text_image, (0, base_height))
            elif position == 'top':
                # 拼接在顶部
                new_image = Image.new('RGB', (base_width, base_height + text_height), (255, 255, 255))
                new_image.paste(text_image, (0, 0))
                new_image.paste(base_image, (0, text_height))
            elif position == 'right':
                # 拼接在右侧
                new_image = Image.new('RGB', (base_width + text_width, base_height), (255, 255, 255))
                new_image.paste(base_image, (0, 0))
                new_image.paste(text_image, (base_width, 0))
            elif position == 'left':
                # 拼接在左侧
                new_image = Image.new('RGB', (base_width + text_width, base_height), (255, 255, 255))
                new_image.paste(text_image, (0, 0))
                new_image.paste(base_image, (text_width, 0))
            
            return new_image

        # Define font settings
        font_folder = "fonts"
        font_file = os.path.join(font_folder, font_name)
        resolved_font_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), font_file)
        font_size = 40

        # text_image = create_text_image(text, resolved_font_path, font_size, width=width, height=height, position=position)
        # transform = transforms.ToTensor()
        # # 转换图片为Tensor
        # tensor_image = transform(text_image)

        result_image = []
        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            print(image.shape)
            # 创建文字图片
            text_image = create_text_image(text, resolved_font_path, font_size, width=image.shape[1], height=image.shape[0], position=position)

            # 拼接文字图片到目标图片的底部
            result_image.append(pil2tensor(concatenate_images(img, text_image, position=position)))

        return  (torch.cat(result_image, dim=0), )

folder_paths.folder_names_and_paths["aesthetic"] = ([os.path.join(folder_paths.models_dir,"aesthetic")], folder_paths.supported_pt_extensions)

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating', batch_norm=True):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048) if batch_norm else nn.Identity(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512) if batch_norm else nn.Identity(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256) if batch_norm else nn.Identity(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128) if batch_norm else nn.Identity(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

class ImageScorer:
    def __init__(self):
        self.model = None
        self.model2 = None
        self.preprocess = None
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("aesthetic"),),
                "image": ("IMAGE",),
                "device": ("STRING", {
                    "multiline": False,
                    "default": "cuda"
                }),
            }
        }

    RETURN_TYPES = ("NUMBER","FLOAT","STRING")
    FUNCTION = "calc_score"
    CATEGORY = "SelfNodes/图片"

    def calc_score(self, model_name, image, device):
        m_path = folder_paths.folder_names_and_paths["aesthetic"][0]
        m_path2 = os.path.join(m_path[0], model_name)
        if self.model is None:
            self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
            s = torch.load(m_path2, weights_only=False)
            self.model.load_state_dict(s)
            self.model.to(device)
            self.model.eval()
        if self.model2 is None:
            self.model2, self.preprocess = clip.load("ViT-L/14", device=device)  # RN50x64
        tensor_image = image[0]
        img = (tensor_image * 255).to(torch.uint8).numpy()
        pil_image = Image.fromarray(img, mode='RGB')
        image2 = self.preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = self.model2.encode_image(image2)
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = self.model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
        final_prediction = round(float(prediction[0]), 2)
        return (final_prediction,final_prediction,str(final_prediction),)


def load_image_and_mask_from_url(url, timeout=10):
    # Load the image from the URL
    response = requests.get(url, timeout=timeout)

    content_type = response.headers.get('Content-Type')
    
    image = Image.open(BytesIO(response.content))

    # Create a mask from the image's alpha channel
    mask = image.convert('RGBA').split()[-1]

    # Convert the mask to a black and white image
    mask = mask.convert('L')

    image=image.convert('RGB')

    return (image, mask)

class LoadImagesFromURL:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "url": ("STRING",{"multiline": True,"default": "https://","dynamicPrompts": False}),
                             },
                "optional":{
                    "seed": (any_type,  {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                }
                }
    
    RETURN_TYPES = ("IMAGE","MASK",)
    RETURN_NAMES = ("images","masks",)

    FUNCTION = "run"

    CATEGORY = "SelfNodes/图片"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True,True,)


    global urls_image
    urls_image={}

    def run(self,url,seed=0):
        global urls_image
        # print(urls_image)
        def filter_http_urls(urls):
            filtered_urls = []
            for url in urls.split('\n'):
                if url.startswith('http'):
                    filtered_urls.append(url)
            return filtered_urls

        filtered_urls = filter_http_urls(url)

        images=[]
        masks=[]

        for img_url in filtered_urls:
            try:
                if img_url in urls_image:
                    img,mask=urls_image[img_url]
                else:
                    img,mask=load_image_and_mask_from_url(img_url)
                    urls_image[img_url]=(img,mask)

                img1=pil2tensor(img)
                mask1=pil2tensor(mask)

                images.append(img1)
                masks.append(mask1)
            except Exception as e:
                print("发生了一个未知的错误：", str(e))

        return (images,masks,)

class SelfNodes_LoadImagesDIr(object):
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
                "load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": -1, "min": -1, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")

    FUNCTION = "load_images"

    CATEGORY = "SelfNodes/图片"

    def load_images(self, seed, directory: str, load_cap: int = -1, start_index: int = 0):
        if not os.path.isdir(directory):
            dir_files = [directory]
            # raise FileNotFoundError(f"Directory '{directory} cannot be found.'")
        else:
            dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        random.seed(seed)

        if os.path.isdir(directory):
            # Filter files by extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
    
            dir_files = sorted(dir_files)
            dir_files = [os.path.join(directory, x) for x in dir_files]

        # start at start_index
        if start_index>=0 and dir_files[start_index:]:
            dir_files = dir_files[start_index:]
        else:
            dir_files = dir_files[random.randint(0, len(dir_files)-1):]

        file_name = os.path.basename(dir_files[0])
        images = []
        masks = []

        limit_images = False
        if load_cap > 0:
            limit_images = True
        image_count = 0

        has_non_empty_mask = False

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
                has_non_empty_mask = True
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)
            image_count += 1

        if len(images) == 1:
            return (images[0], masks[0], 1, file_name)

        elif len(images) > 1:
            image1 = images[0]
            mask1 = None

            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)

            for mask2 in masks:
                if has_non_empty_mask:
                    if image1.shape[1:3] != mask2.shape:
                        mask2 = torch.nn.functional.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(image1.shape[1], image1.shape[2]), mode='bilinear', align_corners=False)
                        mask2 = mask2.squeeze(0)
                    else:
                        mask2 = mask2.unsqueeze(0)
                else:
                    mask2 = mask2.unsqueeze(0)

                if mask1 is None:
                    mask1 = mask2
                else:
                    mask1 = torch.cat((mask1, mask2), dim=0)

        return (image1, mask1, len(images), file_name)


class SelfNodes_JoinImages(object):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "width": ("INT", {"default": 1024, "min": 64, "max": 2048}),
                # "height": ("INT", {"default": 1024, "min": 64, "max": 2048}),
                "images1": ("IMAGE", ),
                "images2": ("IMAGE", ),
                "position": (["top", "bottom", "left", "right", "overlay"],  {"default": "top"}),
                "fg_offset_X": ("INT", {"default": 0, "max": 0xffffffffffffffff}),
                "fg_offset_Y": ("INT", {"default": 0, "max": 0xffffffffffffffff}),
            },
        }


    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "join_images"

    CATEGORY = "SelfNodes/图片"

    def join_images(self, images1, images2, position, fg_offset_X, fg_offset_Y,):
        def concatenate_images(base_image, join_image, position='bottom'):
            # 获取拼接后图像的尺寸
            base_width, base_height = base_image.size
            join_width, join_height = join_image.size

            if position == 'bottom':
                # 拼接在底部
                new_image = Image.new('RGB', (base_width, base_height + join_height), (255, 255, 255))
                new_image.paste(base_image, (0, 0))
                new_image.paste(join_image, (0, base_height))
            elif position == 'top':
                # 拼接在顶部
                new_image = Image.new('RGB', (base_width, base_height + join_height), (255, 255, 255))
                new_image.paste(join_image, (0, 0))
                new_image.paste(base_image, (0, join_height))
            elif position == 'right':
                # 拼接在右侧
                new_image = Image.new('RGB', (base_width + join_width, base_height), (255, 255, 255))
                new_image.paste(base_image, (0, 0))
                new_image.paste(join_image, (base_width, 0))
            elif position == 'left':
                # 拼接在左侧
                new_image = Image.new('RGB', (base_width + join_width, base_height), (255, 255, 255))
                new_image.paste(join_image, (0, 0))
                new_image.paste(base_image, (join_width, 0))
            elif position == 'overlay':
                # 叠加
                max_base_width = max(base_width, join_width)
                max_base_height = max(base_height, join_height)
                new_image = Image.new('RGBA', (max_base_width+abs(fg_offset_X), max_base_height+abs(fg_offset_Y)), (255, 255, 255))
                centre_x = int(max_base_width/2)
                centre_y = int(max_base_height/2)
                # Image.alpha_composite(base_image)
                new_image.paste(base_image, (centre_x-int(base_width/2), centre_y-int(base_height/2)), base_image)
                new_image.paste(join_image, (centre_x-int(join_width/2)+fg_offset_X, centre_y-int(join_height/2)+fg_offset_Y), join_image)
                new_image = new_image.convert("RGB")

            return new_image

        # Define font settings
        result_image = []
        for batch_number, image in enumerate(images1):
            i1 = 255. * image.cpu().numpy()
            img1 = Image.fromarray(np.clip(i1, 0, 255).astype(np.uint8))

            i2 = 255. * images2[0].cpu().numpy()
            img2 = Image.fromarray(np.clip(i2, 0, 255).astype(np.uint8))

            result_image.append(pil2tensor(concatenate_images(img1, img2, position=position)))

        return  (torch.cat(result_image, dim=0), )

class SelfNodes_PersonSplit(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "level": (["n", "s", "m", "x"], {"default": "m"}),
                "conf_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "iou_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "person_split"
    CATEGORY = "SelfNodes/图片"

    def person_split(self, images, level: str = 'm', conf_threshold: float = 0.3, iou_threshold: float = 0.5, ):
        result_image = []
        for batch_number, image in enumerate(images):
            i1 = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i1, 0, 255).astype(np.uint8))

            detection = detect_person(img, level, 'v1.1', conf_threshold=conf_threshold, iou_threshold=iou_threshold)

            for i, (area, type_, score) in enumerate(detection):
                result_image.append(pil2tensor(img.crop(area)))

        return  (torch.cat(result_image, dim=0), )


NODE_CLASS_MAPPINGS = {
    "保存图片JPG": SaveImageJPG,
    "保存图片WEBP": SaveImageWEBP,
    "保存图片不预览": SaveImageNotPreview,
    "合并文字图片": TextImg,
    "图片美学评分": ImageScorer,
    "从URL加载图片": LoadImagesFromURL,
    "文件夹加载图片": SelfNodes_LoadImagesDIr,
    "拼接图片": SelfNodes_JoinImages,
    "裁剪人物主体": SelfNodes_PersonSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "保存图片JPG": "保存图片JPG",
    "保存图片WEBP": "保存图片WEBP",
    "保存图片不预览": "保存图片不预览",
    "合并文字图片": "合并文字图片",
    "图片美学评分": "图片美学评分",
    "从URL加载图片": "从URL加载图片",
    "文件夹加载图片": "文件夹加载图片",
    "裁剪人物主体": "裁剪人物主体",
}