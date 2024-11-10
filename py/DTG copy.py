# import os
# import torch

# import re

# import random
# from functools import lru_cache

# import torch

# from transformers import set_seed

# from collections import defaultdict

# from .kgen.formatter import seperate_tags, apply_format, apply_dtg_prompt
# from .kgen.metainfo import TARGET
# from .kgen.generate import tag_gen
# from .kgen.logging import logger

# #Set model dir
# ext_dir = os.path.dirname(os.path.realpath(__file__))
# all_model_file = [f for f in os.listdir(ext_dir + "/models") if f.endswith(".gguf")]

# #Find gguf model
# try:
#     from llama_cpp import Llama, LLAMA_SPLIT_MODE_NONE

#     text_model = Llama(
#         f"{ext_dir}\models\{all_model_file[-1]}",
#         n_ctx=384,
#         split_mode=LLAMA_SPLIT_MODE_NONE,
#         n_gpu_layers=100,
#         verbose=False,
#     )
#     tokenizer = None
# except Exception as e:
#     # 捕获其他所有异常
#     print(f"发生了其他错误: {e}")
#     logger.warning("Llama-cpp-python not found, using transformers to load model")
#     from transformers import LlamaForCausalLM, LlamaTokenizer

#     text_model = (
#         LlamaForCausalLM.from_pretrained("KBlueLeaf/DanTagGen-beta").eval().half()
#     )
#     tokenizer = LlamaTokenizer.from_pretrained("KBlueLeaf/DanTagGen-beta")
#     if torch.cuda.is_available():
#         text_model = text_model.cuda()
#     else:
#         text_model = text_model.cpu()

# #List
# TOTAL_TAG_LENGTH = {
#     "VERY_SHORT": "very short",
#     "SHORT": "short",
#     "LONG": "long",
#     "VERY_LONG": "very long",
# }

# TOTAL_TAG_LENGTH_TAGS = {
#     TOTAL_TAG_LENGTH["VERY_SHORT"]: "<|very_short|>",
#     TOTAL_TAG_LENGTH["SHORT"]: "<|short|>",
#     TOTAL_TAG_LENGTH["LONG"]: "<|long|>",
#     TOTAL_TAG_LENGTH["VERY_LONG"]: "<|very_long|>",
# }

# PROCESSING_TIMING = {
#     "BEFORE": "Before applying other prompt processings",
#     "AFTER": "After applying other prompt processings",
# }

# re_attention = re.compile(r"""
# \\\(|
# \\\)|
# \\\[|
# \\]|
# \\\\|
# \\|
# \(|
# \[|
# :\s*([+-]?[.\d]+)\s*\)|
# \)|
# ]|
# [^\\()\[\]:]+|
# :
# """, re.X)

# re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

# def parse_prompt_attention(text):
#     """
#     Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
#     Accepted tokens are:
#       (abc) - increases attention to abc by a multiplier of 1.1
#       (abc:3.12) - increases attention to abc by a multiplier of 3.12
#       [abc] - decreases attention to abc by a multiplier of 1.1
#       \( - literal character '('
#       \[ - literal character '['
#       \) - literal character ')'
#       \] - literal character ']'
#       \\ - literal character '\'
#       anything else - just text

#     >>> parse_prompt_attention('normal text')
#     [['normal text', 1.0]]
#     >>> parse_prompt_attention('an (important) word')
#     [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
#     >>> parse_prompt_attention('(unbalanced')
#     [['unbalanced', 1.1]]
#     >>> parse_prompt_attention('\(literal\]')
#     [['(literal]', 1.0]]
#     >>> parse_prompt_attention('(unnecessary)(parens)')
#     [['unnecessaryparens', 1.1]]
#     >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
#     [['a ', 1.0],
#      ['house', 1.5730000000000004],
#      [' ', 1.1],
#      ['on', 1.0],
#      [' a ', 1.1],
#      ['hill', 0.55],
#      [', sun, ', 1.1],
#      ['sky', 1.4641000000000006],
#      ['.', 1.1]]
#     """

#     res = []
#     round_brackets = []
#     square_brackets = []

#     round_bracket_multiplier = 1.1
#     square_bracket_multiplier = 1 / 1.1

#     def multiply_range(start_position, multiplier):
#         for p in range(start_position, len(res)):
#             res[p][1] *= multiplier

#     for m in re_attention.finditer(text):
#         text = m.group(0)
#         weight = m.group(1)

#         if text.startswith('\\'):
#             res.append([text[1:], 1.0])
#         elif text == '(':
#             round_brackets.append(len(res))
#         elif text == '[':
#             square_brackets.append(len(res))
#         elif weight is not None and round_brackets:
#             multiply_range(round_brackets.pop(), float(weight))
#         elif text == ')' and round_brackets:
#             multiply_range(round_brackets.pop(), round_bracket_multiplier)
#         elif text == ']' and square_brackets:
#             multiply_range(square_brackets.pop(), square_bracket_multiplier)
#         else:
#             parts = re.split(re_break, text)
#             for i, part in enumerate(parts):
#                 if i > 0:
#                     res.append(["BREAK", -1])
#                 res.append([part, 1.0])

#     for pos in round_brackets:
#         multiply_range(pos, round_bracket_multiplier)

#     for pos in square_brackets:
#         multiply_range(pos, square_bracket_multiplier)

#     if len(res) == 0:
#         res = [["", 1.0]]

#     # merge runs of identical weights
#     i = 0
#     while i + 1 < len(res):
#         if res[i][1] == res[i + 1][1]:
#             res[i][0] += res[i + 1][0]
#             res.pop(i + 1)
#         else:
#             i += 1

#     return res

class SelfNodesDanTagGen1:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "ban_tags": ("STRING", {"default": "", "multiline": True}),
                "format": ("STRING", {"default": "<|special|>, \n <|characters|>, <|copyrights|>, \n <|artist|>, \n <|general|>, \n <|quality|>, <|meta|>, <|rating|>", "multiline": True}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "temperature": ("FLOAT", {"default": 1.35, "step": 0.01}),
                # "tag_length": (["very_short", "short", "long", "very_long"], {"default":"long"}),
                "seed": ("INT", {"default": 1234}),
            },
        }

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('prompt',)
    FUNCTION = 'execute'
    CATEGORY = "SelfNodes"
    
    def execute(
        self,
        prompt: str,
        # width : int,
        # height : int,
        # seed: int,
        # tag_length: str,
        # ban_tags: str,
        # format: str,
        # temperature: float,
    ):
        # set_seed(seed)
        
        # aspect_ratio = width / height
        
        # prompt_without_extranet = prompt
        # #res = defaultdict(list)
        # prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)

        # # rebuild_extranet = ""
        # # for name, params in res.items():
        #     # for param in params:
        #         # items = ":".join(param.items)
        #         # rebuild_extranet += f" <{name}:{items}>"

        # black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
        # all_tags = []
        # strength_map = {}
        # for part, strength in prompt_parse_strength:
        #     part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
        #     all_tags.extend(part_tags)
        #     if strength == 1:
        #         continue
        #     for tag in part_tags:
        #         strength_map[tag] = strength

        # tag_length = tag_length.replace(" ", "_")
        # len_target = TARGET[tag_length]

        # tag_map = seperate_tags(all_tags)
        # dtg_prompt = apply_dtg_prompt(tag_map, tag_length, aspect_ratio)
        # for llm_gen, extra_tokens in tag_gen(
        #     text_model,
        #     tokenizer,
        #     dtg_prompt,
        #     tag_map["special"] + tag_map["general"],
        #     len_target,
        #     black_list,
        #     temperature=temperature,
        #     top_p=0.95,
        #     top_k=100,
        #     max_new_tokens=256,
        #     max_retry=5,
        # ):
        #     pass
        # tag_map["general"] += extra_tokens
        # for cate in tag_map.keys():
        #     new_list = []
        #     for tag in tag_map[cate]:
        #         tag = tag.replace("(", "\(").replace(")", "\)")
        #         if tag in strength_map:
        #             new_list.append(f"({tag}:{strength_map[tag]})")
        #         else:
        #             new_list.append(tag)
        #     tag_map[cate] = new_list
        # prompt_by_dtg = apply_format(tag_map, format)
        # #return prompt_by_dtg + "\n" + rebuild_extranet
        # print(prompt_by_dtg)
        return ("prompt_by_dtg",)

NODE_CLASS_MAPPINGS = {
    "SelfNodesDanTagGen1": SelfNodesDanTagGen1
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfNodesDanTagGen1": "SelfNodesDanTagGen1"
}
