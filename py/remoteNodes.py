# ------------------------------------------------------------------------#
# 远程执行节点 (跨 ComfyUI 实例)
#
# 用途: 把本机一部分消耗性能的节点搬到另一台 ComfyUI 上执行。
#
#   节点1 SelfNodes_RemoteInput  -> 放在【远程 ComfyUI B】的工作流开头
#       它把 A 注入过来的文本/图片原样"返回"给 B 里的重节点。
#       文本直接透传; 图片则按 A 上传后的文件名从 B 的 input 目录加载成 tensor。
#
#   节点2 SelfNodes_RemoteRequest -> 放在【本地 ComfyUI A】的工作流中
#       替换掉那部分重节点。它把 A 的输入数据上传/注入到远程工作流 JSON,
#       提交给 B 的 /prompt, 轮询 /history 等 B 跑完, 再取出图片/文本结果返回。
#
# 远程工作流模板: 在 B 的 UI 里自己搭好(开头放 RemoteInput 节点, 结尾放
# SaveImage / 文本输出节点), 然后用 ComfyUI 的 "导出(API格式)" 保存为 json。
# 把该 json 的路径(或内容)填给 RemoteRequest 节点的 workflow_path/workflow。
#
# 只有图片和文字跨机器传输; 模型/CLIP 权重无法序列化, 必须由 B 自己加载。
# ------------------------------------------------------------------------#

import json
import os
import time

import requests
import numpy as np
import torch
from PIL import Image
from io import BytesIO

try:
    import folder_paths
except Exception:
    folder_paths = None


class AnyType(str):
    """A special type that can be connected to any other types."""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")

# 无有效图片时的占位(1x1 黑图), 避免下游节点收到空 tensor 而崩溃
_BLANK_IMAGE = None


def _blank_image():
    global _BLANK_IMAGE
    if _BLANK_IMAGE is None:
        _BLANK_IMAGE = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    return _BLANK_IMAGE


def _tensor_to_pil(image):
    """IMAGE tensor (N,H,W,C float 0-1) -> PIL.Image (取第一帧)."""
    if image is None:
        return None
    i = 255.0 * image.cpu().numpy()
    i = np.clip(i, 0, 255).astype(np.uint8)
    if i.ndim == 4:
        i = i[0]
    return Image.fromarray(i)


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


def _load_local_image(image_name):
    """从本机(input 目录)按文件名加载图片 -> IMAGE tensor."""
    if not image_name:
        return _blank_image()
    if folder_paths is None:
        return _blank_image()
    path = os.path.join(folder_paths.get_input_directory(), image_name)
    if not os.path.exists(path):
        print(f"[selfNodes][远程] 找不到图片: {path}")
        return _blank_image()
    try:
        rgb = Image.open(path).convert("RGB")
        return _pil_to_tensor(rgb)
    except Exception as e:
        print(f"[selfNodes][远程] 读取图片失败 {image_name}: {e}")
        return _blank_image()


def _upload_image(url, pil, name):
    """把 PIL 图片以 multipart 字段 image 上传到远程, 返回 {'name','subfolder','type'}."""
    buffer = BytesIO()
    pil.save(buffer, format="PNG")
    buffer.seek(0)
    resp = requests.post(
        f"{url}/upload/image",
        files={"image": (name, buffer, "image/png")},
        data={"overwrite": "true"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


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
        raise RuntimeError(
            f"远程 /prompt 返回 {resp.status_code}: {resp.text[:500]}"
        )
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

            # status 存在但既非 success 也非 error (可能是中间态) -> 继续等
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


def _lookup_remote_input_node(prompt_obj, title):
    """按 _meta.title 或 class_type 找到远程输入节点, 返回 (node_id, node)."""
    found = []
    for node_id, node in prompt_obj.items():
        meta_title = (node.get("_meta") or {}).get("title")
        if meta_title == title or node.get("class_type") == "SelfNodes_RemoteInput":
            found.append((node_id, node))
    return found


def _inject_strings(prompt_obj, title, text_1, text_2):
    """把文本注入到远程输入节点的 text_1 / text_2."""
    for node_id, node in _lookup_remote_input_node(prompt_obj, title):
        node.setdefault("inputs", {})["text_1"] = text_1
        node.setdefault("inputs", {})["text_2"] = text_2
        print(f"[selfNodes][远程] 注入文本 -> 节点 {node_id}")


def _inject_images(prompt_obj, title, url, image_1, image_2):
    """上传 IMAGE 到远程并注入对应文件名."""
    for node_id, node in _lookup_remote_input_node(prompt_obj, title):
        node.setdefault("inputs", {})
        if image_1 is not None:
            pil = _tensor_to_pil(image_1)
            info = _upload_image(url, pil, "remote_input_1.png")
            node["inputs"]["image_1"] = info["name"]
            sub = info.get("subfolder", "")
            print(f"[selfNodes][远程] 上传图片1 -> {info['name']} (subfolder={sub})")
        if image_2 is not None:
            pil = _tensor_to_pil(image_2)
            info = _upload_image(url, pil, "remote_input_2.png")
            node["inputs"]["image_2"] = info["name"]
            print(f"[selfNodes][远程] 上传图片2 -> {info['name']}")


def _inject_seed(prompt_obj, seed):
    """把 seed 覆盖到远程工作流里所有含 seed 的节点(未连接的端)."""
    if seed is None or seed < 0:
        return
    injected = 0
    for node_id, node in prompt_obj.items():
        inputs = node.get("inputs") or {}
        if "seed" in inputs and not isinstance(inputs["seed"], list):
            inputs["seed"] = seed
            injected += 1
    if injected:
        print(f"[selfNodes][远程] 覆盖 {injected} 个节点的 seed = {seed}")


def _collect_results(entry, url):
    """从 history 条目提取图片与文本结果.

    返回 (images:list[tensor], texts:list[str]).
    图片跳过 temp 预览, 只取 output 类型.
    """
    images = []
    texts = []
    outputs = (entry or {}).get("outputs", {}) or {}
    for node_id, out in outputs.items():
        if not isinstance(out, dict):
            continue

        for img in out.get("images", []) or []:
            ftype = img.get("type")
            if ftype == "temp":
                continue
            try:
                tensor = _download_image(
                    url,
                    img["filename"],
                    img.get("subfolder", ""),
                    ftype or "output",
                )
                images.append(tensor)
            except Exception as e:
                print(f"[selfNodes][远程] 下载图片失败: {e}")

        for key in ("text", "string", "value"):
            if key in out:
                val = out[key]
                if isinstance(val, list):
                    for item in val:
                        if item is not None:
                            texts.append(str(item))
                elif val is not None:
                    texts.append(str(val))
    return images, texts


# ------------------------------------------------------------------------#
class SelfNodes_RemoteInput:
    """远程输入(放在 ComfyUI B 工作流开头).

    接收 A 注入过来的文本与图片文件名, 原样返回给 B 内的后续节点.
    图片按文件名从 B 的 input 目录加载.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"multiline": True, "default": ""}),
                "text_2": ("STRING", {"multiline": True, "default": ""}),
                "image_1": ("STRING", {"default": "", "tooltip": "A 上传到本机的图片文件名"}),
                "image_2": ("STRING", {"default": "", "tooltip": "A 上传到本机的图片文件名"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("文本1", "文本2", "图片1", "图片2")
    FUNCTION = "output"
    CATEGORY = "SelfNodes/远程"

    def output(self, text_1, text_2, image_1, image_2):
        img_1 = _load_local_image(image_1)
        img_2 = _load_local_image(image_2)
        return (text_1, text_2, img_1, img_2)


# ------------------------------------------------------------------------#
class SelfNodes_RemoteRequest:
    """请求远程 ComfyUI(放在本地 ComfyUI A 的工作流中).

    上传本机图片 -> 把文本/图片注入远程工作流 JSON -> 提交 /prompt ->
    轮询 /history -> 取回远程输出的图片和文本结果返回给本机后续节点.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_url": ("STRING", {"default": "http://127.0.0.1:8188"}),
                "workflow_path": ("STRING", {"default": "", "tooltip": "远程工作流(API格式 json)的文件路径"}),
                "remote_input_title": ("STRING", {"default": "远程输入", "tooltip": "远程工作流里 RemoteInput 节点的标题"}),
                "text_1": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "text_2": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF, "tooltip": ">=0 覆盖远程含 seed 的节点; -1 保持模板不变"}),
                "timeout": ("INT", {"default": 600, "min": 30, "max": 86400}),
            },
            "optional": {
                "workflow": ("STRING", {"multiline": True, "default": "", "tooltip": "远程工作流 JSON 内容(与 workflow_path 二选一)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("图片1", "图片2", "文本1", "文本2")
    FUNCTION = "request"
    CATEGORY = "SelfNodes/远程"

    def request(
        self,
        remote_url,
        workflow_path,
        remote_input_title,
        text_1,
        text_2,
        image_1,
        image_2,
        seed,
        timeout,
        workflow="",
    ):
        url = _norm_url(remote_url)

        # 1. 载入远程工作流(API 格式)
        wf_text = workflow
        if not wf_text and workflow_path:
            with open(workflow_path, "r", encoding="utf-8") as f:
                wf_text = f.read()
        if not wf_text:
            raise ValueError("请提供 workflow(JSON 内容)或 workflow_path(API 格式 json 文件路径)")
        prompt_obj = json.loads(wf_text)

        # 2. 注入文本与图片
        _inject_strings(prompt_obj, remote_input_title, text_1, text_2)
        _inject_images(prompt_obj, remote_input_title, url, image_1, image_2)
        _inject_seed(prompt_obj, seed)

        # 3. 提交执行
        prompt_id = _submit_prompt(url, prompt_obj)
        print(f"[selfNodes][远程] 已提交, prompt_id={prompt_id}")

        # 4. 轮询等待完成
        entry = _poll_history(url, prompt_id, timeout=timeout)

        # 5. 取回图片与文本结果
        images, texts = _collect_results(entry, url)

        img_1 = images[0] if len(images) > 0 else _blank_image()
        img_2 = images[1] if len(images) > 1 else _blank_image()
        t_1 = texts[0] if len(texts) > 0 else ""
        t_2 = texts[1] if len(texts) > 1 else ""
        return (img_1, img_2, t_1, t_2)


# ------------------------------------------------------------------------#
# MAPPINGS
# ------------------------------------------------------------------------#
NODE_CLASS_MAPPINGS = {
    "SelfNodes Remote Input": SelfNodes_RemoteInput,
    "SelfNodes Remote Request": SelfNodes_RemoteRequest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfNodes Remote Input": "远程输入",
    "SelfNodes Remote Request": "请求远程ComfyUI",
}
