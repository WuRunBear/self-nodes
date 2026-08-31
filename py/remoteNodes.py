# ------------------------------------------------------------------------#
# 远程执行节点 (跨 ComfyUI 实例)
#
# 用途: 把本机一部分消耗性能的节点搬到另一台 ComfyUI 上执行。
#
#   请求远程ComfyUI (放本地 A):
#     1. 接受远程工作流 JSON (API 格式): 在 B 的 UI 里搭好后
#        "导出(API格式)", 把 JSON 内容粘进 workflow 或填文件路径。
#        工作流本身就是参数表 —— 想改任何参数, 直接改 JSON 里
#        任意节点的任意输入值。
#     2. 可选"参数JSON"注入动态值, 格式:
#            {"节点标题或ID": {"输入名": 值}}
#        例: {"提示词": {"text": "一只猫"}, "3": {"seed": 123},
#             "9": {"image": {"$file": "/path/to/local.png"}}}
#        值为 {"$file": "本地文件路径"} 时自动上传该文件并注入文件名。
#     3. 提交到远程 /prompt, 轮询 /history, 取回全部输出:
#        图片(合并为一个 batch) + 文本(换行拼接)。
#
#   远程文本输出 (放远程 B, 可选): OUTPUT_NODE 节点, 把文字写进
#     history 供"请求远程ComfyUI"取回; 图片结果用标准 SaveImage 即可。
#
# 远程 B 只需标准 ComfyUI (要取回文字结果时才需装本扩展)。
# 模型/CLIP 权重无法序列化, 必须由 B 自己加载; 跨机器传输的只有
# 图片文件与文本。
# ------------------------------------------------------------------------#

import json
import os
import time

import requests
import numpy as np
import torch
from PIL import Image
from io import BytesIO


def _blank_image():
    """无结果时的占位(1x1 黑图), 避免下游节点收到空 tensor 而崩溃."""
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _pil_to_tensor(pil):
    """PIL.Image -> IMAGE tensor (1,H,W,C)."""
    return torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).unsqueeze(0)


def _norm_url(url):
    """规范化远程地址, 补 http:// 并去掉末尾斜杠."""
    url = (url or "").strip()
    if not url:
        raise ValueError("远程地址为空 (remote_url)")
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    return url.rstrip("/")


def _upload_file(url, path):
    """上传本地文件到远程, 返回远程保存的文件名."""
    if not os.path.isfile(path):
        raise ValueError(f"文件不存在: {path}")
    name = os.path.basename(path)
    with open(path, "rb") as f:
        resp = requests.post(
            f"{url}/upload/image",
            files={"image": (name, f)},
            data={"overwrite": "true"},
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()["name"]


def _submit_prompt(url, prompt_obj, client_id="selfnodes"):
    """提交工作流(API 格式), 返回 prompt_id."""
    data = {"prompt": prompt_obj, "client_id": client_id}
    resp = requests.post(
        f"{url}/prompt",
        data=json.dumps(data),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"远程 /prompt 返回 {resp.status_code}: {resp.text[:500]}")
    resp_json = resp.json()
    prompt_id = resp_json.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"没有返回 prompt_id: {resp_json}")
    return prompt_id


def _poll_history(url, prompt_id, timeout=600, interval=1.0):
    """轮询 /history/{prompt_id} 直到远程执行结束. 返回 history 条目."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/history/{prompt_id}", timeout=30)
            resp.raise_for_status()
            history = resp.json()
        except requests.RequestException as e:
            print(f"[selfNodes][远程] 轮询失败, 重试: {e}")
            time.sleep(interval)
            continue

        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {}) or {}
            status_str = status.get("status_str")

            if status_str == "error":
                msgs = status.get("messages", [])
                raise RuntimeError(f"远程执行失败: {msgs}")

            if status_str == "success":
                return entry

            print(f"[selfNodes][远程] 执行中... status={status}")
        time.sleep(interval)

    raise TimeoutError(f"等待远程执行超时 ({timeout}s)")


def _download_image(url, filename, subfolder="", ftype="output"):
    """GET /view 下载远程输出图片 -> IMAGE tensor."""
    params = {"filename": filename, "subfolder": subfolder, "type": ftype}
    resp = requests.get(f"{url}/view", params=params, timeout=60)
    resp.raise_for_status()
    pil = Image.open(BytesIO(resp.content)).convert("RGB")
    return _pil_to_tensor(pil)


def _resolve_targets(prompt_obj, key):
    """参数JSON 的键 -> 目标节点列表. 先按 _meta.title 匹配, 再按节点ID."""
    by_title = []
    for nid, node in prompt_obj.items():
        if not isinstance(node, dict):
            continue
        if (node.get("_meta") or {}).get("title") == key:
            by_title.append((nid, node))
    if by_title:
        return by_title

    node = prompt_obj.get(key)
    if isinstance(node, dict):
        return [(key, node)]
    return None


def _apply_params(url, prompt_obj, params):
    """把参数JSON 注入到工作流的任意节点任意输入.

    格式: {"节点标题或ID": {"输入名": 值}}
    值为 {"$file": "本地路径"} 时上传该文件并注入远程文件名.
    """
    if not isinstance(params, dict):
        raise ValueError("参数JSON 必须是对象, 如 {\"节点标题或ID\": {\"输入名\": 值}}")

    for key, patch in params.items():
        targets = _resolve_targets(prompt_obj, key)
        if not targets:
            titles = [
                (node.get("_meta") or {}).get("title") or nid
                for nid, node in prompt_obj.items()
                if isinstance(node, dict)
            ]
            raise ValueError(f"参数JSON: 找不到节点 '{key}' (可用标题/ID: {titles})")
        if not isinstance(patch, dict):
            raise ValueError(f"参数JSON: 节点 '{key}' 的值必须是对象 {{\"输入名\": 值}}")

        for nid, node in targets:
            inputs = node.setdefault("inputs", {})
            for name, value in patch.items():
                if name not in inputs:
                    print(f"[selfNodes][远程] 警告: 节点 '{key}'({nid}) 没有输入 '{name}', 仍将写入")
                if isinstance(value, dict) and "$file" in value:
                    value = _upload_file(url, value["$file"])
                inputs[name] = value
            print(f"[selfNodes][远程] 已注入参数 -> 节点 {nid} ({key}): {list(patch)}")


def _collect_results(entry, url):
    """从 history 条目提取全部图片与文本.

    图片跳过 temp 预览; 文本兼容 text/string/value 键(值为数组).
    返回 (images: list[tensor], texts: list[str]).
    """
    images = []
    texts = []
    outputs = (entry or {}).get("outputs", {}) or {}
    for node_id, out in outputs.items():
        if not isinstance(out, dict):
            continue

        for img in out.get("images", []) or []:
            if img.get("type") == "temp":
                continue
            try:
                images.append(
                    _download_image(
                        url,
                        img["filename"],
                        img.get("subfolder", ""),
                        img.get("type") or "output",
                    )
                )
            except Exception as e:
                print(f"[selfNodes][远程] 下载图片失败: {e}")

        for key in ("text", "string", "value"):
            if key not in out:
                continue
            val = out[key]
            if isinstance(val, list):
                texts.extend(str(item) for item in val if item is not None)
            elif val is not None:
                texts.append(str(val))
    return images, texts


# ------------------------------------------------------------------------#
class SelfNodes_RemoteRequest:
    """请求远程 ComfyUI (放本地 A 的工作流中).

    接受远程工作流 JSON + 可选参数JSON (注入任意节点的任意输入),
    提交远程执行并取回图片/文本结果.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_url": ("STRING", {"default": "http://127.0.0.1:8188"}),
                "workflow": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "远程工作流 JSON (API格式), 参数直接改在这里",
                }),
                "params_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": '可选: {"节点标题或ID": {"输入名": 值}}; 值支持 {"$file": "本地文件路径"}',
                }),
                "timeout": ("INT", {"default": 600, "min": 30, "max": 86400}),
            },
            "optional": {
                "workflow_path": ("STRING", {"default": "", "tooltip": "从文件读取工作流 JSON (workflow 为空时生效)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图片", "文本")
    FUNCTION = "request"
    CATEGORY = "SelfNodes/远程"

    def request(self, remote_url, workflow, params_json, timeout, workflow_path=""):
        url = _norm_url(remote_url)

        # 1. 载入远程工作流(API 格式)
        wf_text = workflow
        if not wf_text and workflow_path:
            with open(workflow_path, "r", encoding="utf-8") as f:
                wf_text = f.read()
        if not wf_text:
            raise ValueError("请提供 workflow (JSON 内容) 或 workflow_path (API 格式 json 文件路径)")
        try:
            prompt_obj = json.loads(wf_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"工作流 JSON 解析失败: {e}")

        if isinstance(prompt_obj, dict) and "nodes" in prompt_obj and "links" in prompt_obj:
            raise ValueError("这是 UI 格式工作流, 请在远程 ComfyUI 使用 导出(API格式) 后重新提供")

        # 2. 注入参数 (可选)
        if params_json.strip():
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"参数JSON 解析失败: {e}")
            _apply_params(url, prompt_obj, params)

        # 3. 提交执行
        prompt_id = _submit_prompt(url, prompt_obj)
        print(f"[selfNodes][远程] 已提交, prompt_id={prompt_id}")

        # 4. 轮询等待完成
        entry = _poll_history(url, prompt_id, timeout=timeout)

        # 5. 取回图片与文本结果
        images, texts = _collect_results(entry, url)

        if images:
            try:
                image_out = torch.cat(images, dim=0)
            except Exception as e:
                print(f"[selfNodes][远程] 图片尺寸不一致无法合并({e}), 只返回第一张")
                image_out = images[0]
        else:
            image_out = _blank_image()

        return (image_out, "\n".join(texts))


# ------------------------------------------------------------------------#
class SelfNodes_RemoteTextOutput:
    """远程文本输出 (放远程 B 的工作流末尾, 可选).

    OUTPUT_NODE: 把文字写进 history, 供"请求远程ComfyUI"取回.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "output"
    OUTPUT_NODE = True
    CATEGORY = "SelfNodes/远程"

    def output(self, text):
        return {"ui": {"text": [text]}, "result": (text,)}


# ------------------------------------------------------------------------#
# MAPPINGS
# ------------------------------------------------------------------------#
NODE_CLASS_MAPPINGS = {
    "SelfNodes Remote Request": SelfNodes_RemoteRequest,
    "SelfNodes Remote Text Output": SelfNodes_RemoteTextOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfNodes Remote Request": "请求远程ComfyUI",
    "SelfNodes Remote Text Output": "远程文本输出",
}
