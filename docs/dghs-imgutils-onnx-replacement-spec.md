# dghs-imgutils 0.19.0 → onnxruntime 直调替换规格

目标：用 `numpy + Pillow + onnxruntime` 独立复刻 `imgutils.detect` 的
`detect_person / detect_halfbody / detect_heads / detect_faces`，行为与 `dghs-imgutils==0.19.0` **逐位一致**。

> 源码基线：`raw.githubusercontent.com/deepghs/imgutils/main`（提交 `46df848d`, 2025-10-11）对应 PyPI `0.19.0`。
> 验证方式：经 `hf-mirror.com` 下载真实 ONNX，用本规格的独立实现对 `imgutils` 官方单测逐位复现，face/person/head/halfbody **全部命中**（坐标完全相同，score 误差 < 1e-6）。

---

## 0. 实测确认（可直接照抄的关键常量）

| 项目 | 结论 |
|---|---|
| ONNX 输入名 | `images` |
| ONNX 输入 shape | `(batch, 3, H, W)`，dynamic，`float32`，RGB，**值域 0..1** |
| ONNX 输出名 | `output0` |
| ONNX 输出 shape | `(1, 4+nc, N)`，对单人检测 nc=1 → `(1, 5, 8400)` |
| 预处理 | RGB → `PIL.Image.resize((S,S))`（**Pillow 默认 BICUBIC，直接拉伸，无 letterbox / 无 padding**）→ `/255` → `float32` CHW → `[None]` |
| 输入尺寸 S | 这些模型恒为 **640**（详见 §4 的 imgsz 说明） |
| 后处理 | **自写 NMS**（`_yolo_nms`，+1px 面积/交点），**不是** `cv2.dnn.NMSBoxes`（全仓库 grep 无该 API） |
| 坐标还原 | x 按 `old_w/new_w`、y 按 `old_h/new_h` 独立线性缩放，`round()` 后 `int`（即拉伸逆映射） |
| label | 取自 ONNX `custom_metadata_map['names']`：`person` / `face` / `head` / `halfbody` |
| conf 默认 | person 0.3 / halfbody 0.5 / head 0.4 / face 0.25（函数默认值，**会覆盖** threshold.json） |
| iou 默认 | person **0.5**，halfbody/head/face 0.7 |
| threshold.json | 仅当 `conf_threshold=None`（走 general `yolo_predict`）时使用：face 0.307 / person 0.348 / head 0.413 / halfbody 0.577 |

---

## 1. 四个函数 → repo / 目录名 / level 映射

`model_name` 规则（来自 `imgutils/detect/{person,halfbody,head,face}.py`）：

| 函数 | repo_id | model_name 规则 | version 默认 | level 默认 | conf | iou |
|---|---|---|---|---|---|---|
| `detect_person` | `deepghs/anime_person_detection` | `person_detect_{version}_{level}` | `v1.1` | `m` | 0.3 | **0.5** |
| `detect_halfbody` | `deepghs/anime_halfbody_detection` | `halfbody_detect_{version}_{level}` | `v1.0` | `s` | 0.5 | 0.7 |
| `detect_heads` | `deepghs/anime_head_detection` | 默认 `head_detect_v2.0_s`（见下） | — | `None`(≈s) | 0.4 | 0.7 |
| `detect_faces` | `deepghs/anime_face_detection` | `face_detect_{version}_{level}` | `v1.4` | `s` | 0.25 | 0.7 |

level 取值 `n/s/m/l/x`，训练基座为 `yolov8{n,s,m,l,x}.pt`（`zoo/detection/*.py`）。
每个模型的 ONNX 路径固定为 **`{repo_id}/{model_name}/model.onnx`**。

**head 的特殊性（务必注意）**：`detect_heads(image, level=None, model_name='head_detect_v2.0_s', ...)`。
只有 `model_name` 为空时才会用 `head_detect_v0_{level or 's'}`；否则 `model_name or ...` 直接返回默认 `head_detect_v2.0_s`，
**传入的 `level` 被忽略**（仅发 DeprecationWarning）。本仓库 `py/imageNodes.py` 正是位置传参 `detect_heads(img, level, ...)`，等价于永远用 `head_detect_v2.0_s`。

**各 repo 实际存在的目录（来自 HF 文件清单，2024 年最后更新）**：
- face: `v1.4_{n,s}`、`v1.3_{n,s}`、`v1.2_s`、`v1.1_{n,s}`、`v1_{n,s}`、`v0_{n,s}`
- person: `v1.1_{n,s,m}`、`v1.2_s`、`v1.3_s`、`v1_{m,s}`、`v0_{m,s,x}`
- head: `v2.0_{n,s,m,l,x}` 及其 `_yv11` 变体；`v1.6_*`（含 `_yv9/_yv10/_yv11/_rtdetr`）；`v1.0`~`v1.5`；`v0.x`
- halfbody: `v1.0_{n,s}`、`v0.4_s`、`v0.3_s`、`v0.2_s`

> 注意：`detect_heads` 的默认 `head_detect_v2.0_s` 导出带 `model_type.json` / `imgsz=[640,640]`；其余默认模型无这些元数据。

---

## 2. 通用检测流程（`imgutils/generic/yolo.py` 核心，逐句复刻）

### 2.1 预处理
```python
def _image_preprocess(image, max_infer_size=1216, allow_dynamic=False, align=32):
    if isinstance(max_infer_size, int):
        max_infer_width, max_infer_height = max_infer_size, max_infer_size
    else:
        max_infer_width, max_infer_height = max_infer_size
    old_width, old_height = image.width, image.height
    if allow_dynamic:                       # 默认 False，本场景走 else
        r = min(max_infer_width / old_width, max_infer_height / old_height)
        nw = old_width * r if r < 1 else old_width
        nh = old_height * r if r < 1 else old_height
        nw = int(math.ceil(nw / align) * align)
        nh = int(math.ceil(nh / align) * align)
    else:
        nw, nh = max_infer_width, max_infer_height     # 直接拉伸到固定尺寸
    image = image.resize((nw, nh))                     # Pillow 默认 BICUBIC
    return image, (old_width, old_height), (nw, nh)
```
`max_infer_size` 来自 ONNX 元数据 `imgsz`（§4）；缺失时回退整数 `640`。

### 2.2 编码（`imgutils/data/encode.py: rgb_encode`）
```python
def rgb_encode(image, order_='CHW', use_float=True):
    image = load_image(image, mode='RGB')             # RGBA 会先垫白底再转 RGB
    array = np.asarray(image)                          # HWC uint8
    array = np.transpose(array, _get_hwc_map(order_))  # -> CHW
    if use_float:
        array = (array / 255.0).astype(np.float32)
    return array
```
调用：`data = rgb_encode(resized)[None, ...]` → `(1,3,H,W)`。
执行：`output, = sess.run(['output0'], {'images': data})`。

### 2.3 后处理：NMS-based（YOLOv8，本场景唯一路径）
```python
def _yolo_xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def _yolo_nms(boxes, scores, iou_threshold=0.7):       # 自写，非 cv2
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)              # 注意 +1
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1); h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep

def _xy_postprocess(x, y, old_size, new_size):
    x = x / new_size[0] * old_size[0]
    y = y / new_size[1] * old_size[1]
    x = int(np.clip(x, 0, old_size[0]).round())
    y = int(np.clip(y, 0, old_size[1]).round())
    return x, y

def _nms_postprocess(output, conf_threshold, iou_threshold, old_size, new_size, labels):
    assert output.shape[0] == 4 + len(labels)          # output = out[0]，形状 (4+nc, N)
    max_scores = output[4:, :].max(axis=0)
    output = output[:, max_scores > conf_threshold].transpose(1, 0)
    boxes = output[:, :4]                              # cx,cy,w,h
    scores = output[:, 4:]                             # class scores
    filtered_max_scores = scores.max(axis=1)
    if not boxes.size:
        return []
    boxes = _yolo_xywh2xyxy(boxes)
    idx = _yolo_nms(boxes, filtered_max_scores, iou_threshold)
    boxes, scores = boxes[idx], scores[idx]
    detections = []
    for box, score in zip(boxes, scores):
        x0, y0 = _xy_postprocess(box[0], box[1], old_size, new_size)
        x1, y1 = _xy_postprocess(box[2], box[3], old_size, new_size)
        mid = score.argmax()
        detections.append(((x0, y0, x1, y1), labels[mid], float(score[mid])))
    return detections
```

### 2.4 输出分派（`_yolo_postprocess`）
```python
if output.shape[-1] == 6:      # end2end 模型（YOLOv10/v11），[x0,y0,x1,y1,score,cls]
    -> _end2end_postprocess(...)   # 坐标为输入像素空间；NMS 用默认 0.7（忽略传入 iou）
else:                          # YOLOv8 等 NMS-based，[4+nc, N]
    -> _nms_postprocess(...)
```
- RT-DETR（`model_type.json` 为 `rtdetr`）：输出 `(N, 4+nc)`、坐标归一化到 `[0,1]`，走
  `_nms_postprocess(output.transpose(1,0), ..., new_size=(1.0,1.0))`。
- 本替换只覆盖 4 个默认模型（均 YOLOv8，NMS 路径）；若要支持 `head_detect_v1.6_*_yv10/rtdetr`，
  按上面两条分支补即可。`model_type` 默认 `'yolo'`。

---

## 3. 返回值格式

```python
List[Tuple[Tuple[int, int, int, int], str, float]]
# 例如 ((966, 142, 1085, 261), 'face', 0.8504581451416016)
```
- bbox：`(x0, y0, x1, y1)`，**int**，已裁剪到 `[0, W]×[0, H]`，坐标来自 `_xy_postprocess`。
- label：`str`，取 ONNX `names` 映射值（`person`/`face`/`head`/`halfbody`）。
- score：`float`，该框 argmax 类别分数。
- 类型别名（`imgutils/detect/base.py`）：
  `BBoxTyping=Tuple[float,float,float,float]`，`BBoxWithScoreAndLabel=Tuple[BBoxTyping,str,float]`。

---

## 4. 模型下载与旁文件

`imgutils` 用单文件下载（不是 snapshot 目录）：
```python
hf_hub_download(repo_id=repo_id, repo_type='model', filename=f'{model_name}/model.onnx', token=...)
```
读取 ONNX 元数据（`onnxruntime.InferenceSession.get_modelmeta().custom_metadata_map`）：
- `imgsz`：可选，JSON 数组如 `"[640, 640]"` → `max_infer_size`；**缺失则回退 640**。
- `names`：必需，如 `"{0: 'face'}"` → labels。

旁文件（同目录，均**可选**）：
| 文件 | 内容 | 缺失时行为 |
|---|---|---|
| `model_type.json` | `{"model_type": "yolo"\|"rtdetr"}` | 默认 `yolo` |
| `threshold.json` | `{"f1_score":..., "threshold":...}` | 仅在 `conf_threshold=None` 时用；否则用函数默认 |
| `labels.json` | `["face"]` | 不使用（labels 走 ONNX `names`） |
| `model_artifacts.json` | `{"names":..., "yaml":...}` | 不使用 |
| `model.pt` | 训练权重 | 不使用 |

**实测各默认模型元数据**：
- `face_detect_v1.4_s`：仅有 `names={0:'face'}`，**无 imgsz → 640**；threshold 0.307；无 model_type.json。
- `person_detect_v1.1_m`：仅有 `names={0:'person'}`（另有 stride），**无 imgsz → 640**；threshold 0.348。
- `head_detect_v2.0_s`：`names={0:'head'}` + `imgsz=[640,640]` + `task=detect` + `batch=1` + `stride=32`；model_type.json=`yolo`；threshold 0.413。
- `halfbody_detect_v1.0_s`：仅有 `names={0:'halfbody'}`，**无 imgsz → 640**；threshold 0.577。
- 三个 detection repo 的 `lastModified` 均为 2024（模型已稳定，2025-2026 未更新）。

**HF 不可达时的下载方式（实测可用）**：设 `HF_ENDPOINT=https://hf-mirror.com`，或直接
```
https://hf-mirror.com/deepghs/<repo>/resolve/main/<model_name>/model.onnx
```

---

## 5. numpy 2 兼容性判断

对全仓库源码（tarball `deepghs/imgutils@main`）扫描结论：

- **未发现任何 numpy 2.0 已移除的 API**：无 `np.float_/np.complex_/np.unicode_/np.string_`、
  无 `np.NaN/np.Inf/np.infty/np.NINF/np.PINF`、无 `np.float/np.int/np.bool/np.object/np.str/np.long`、
  无 `np.round_/np.product/np.cumproduct/np.alltrue/np.sometrue/np.trapz/np.row_stack/
  np.find_common_type/np.sctypes/np.issubsctype` 等。
- 无 `np.array(..., copy=False)`，无 `numpy.core` 导入。
- 实际使用的 `np.*` 全部在 numpy 2.x 有效：`float128`(有 `hasattr` 守卫)、`bool_`、`floating`、
  `number`、`r_`、`unpackbits`、`argpartition`、`searchsorted`、`insert`、`pad`、`hsplit/hstack`、
  `frombuffer` 等。`imgutils/data/decode.py` 的 `np.float128` 已用 `if hasattr(np,'float128')` 保护。

**判断**：`numpy<2` 是 `requirements.txt`/setup.py 的**保守 pin**，代码本身是 numpy2-clean，`pip install --no-deps` 后强装 `numpy>=2` **大概率可直接 import/运行**（detect 路径）。但要注意：
1. `imgutils` 有硬运行时依赖 `hbutils / hfutils / huggingface_hub / requests`，且 `imgutils.detect` 通过
   `detect/text.py` 传递性 `import cv2`，`imgutils/utils/onnxruntime.py` 会自动安装 onnxruntime —— `--no-deps` 仍需自备这些。
2. 这些下游依赖自身的 numpy2 兼容性另算。
3. 最干净的 numpy2 路线就是本规格的**独立替换**：只依赖 `numpy + Pillow + onnxruntime`，三者均支持 numpy 2。

---

## 6. 参考实现（drop-in，已逐位验证）

见 `docs/anime_detect_standalone.py`。核心管线与 §2 一致；接口签名与 imgutils 保持一致（含默认 conf/iou），
可直接替换 `from imgutils.detect import detect_person, detect_halfbody, detect_heads, detect_faces`。

```python
# 与 imgutils 对齐的默认参数
detect_person   (image, level='m', version='v1.1', model_name=None, conf_threshold=0.3, iou_threshold=0.5)
detect_halfbody (image, level='s', version='v1.0', model_name=None, conf_threshold=0.5, iou_threshold=0.7)
detect_heads    (image, level=None, model_name='head_detect_v2.0_s',        conf_threshold=0.4, iou_threshold=0.7)
detect_faces    (image, level='s', version='v1.4', model_name=None, conf_threshold=0.25, iou_threshold=0.7)
```
