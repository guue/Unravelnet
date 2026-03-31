# UnravelNet 训练与使用



- Backbone 代码：`mmrotate/models/backbones/unravelnet.py`
- 模型注册：`mmrotate/models/backbones/__init__.py`
- 训练配置：`configs/unravelnet/ORCNN_unravelnet_tiny_fpn_le90_dota10val_ss_e36.py`
- 训练配置：`configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py`
- 训练入口：`tools/train.py`
- 测试/评估入口：`tools/test.py`
- 单张图推理：`demo/image_demo.py`
- 大图切片推理：`demo/huge_image_demo.py`

说明：仓库中文件名和代码里使用的是 `UnravelNet` / `unravelnet`，

## 1. 当前仓库里可直接用的 UnravelNet 配置

目前 `configs/unravelnet/` 下只有两个配置：

- `ORCNN_unravelnet_tiny_fpn_le90_dota10val_ss_e36.py`
- `ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py`

它们的共同特点：

- 检测器是 `OrientedRCNN`
- 数据集默认是 `DOTA1.0`
- 输入尺寸是 `1024 x 1024`
- 角度版本是 `le90`
- 训练轮数是 `36` epoch
- 优化器改成了 `AdamW`
- 默认会在训练过程中做验证，并按 `mAP` 保存最佳模型

`tiny` 和 `small` 的主要差别是骨干网络规模不同：

- `tiny`：`stem_dim=32`
- `small`：`stem_dim=64`

## 2. 环境安装

仓库 README 给出的环境是：

```bash
conda create -n LEGNet-Det python=3.8 -y
conda activate LEGNet-Det
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install mmdet
pip install -v -e .
```

虽然环境名写的是 `LEGNet-Det`，但 `UnravelNet` 也是跑在这一套 `MMRotate + MMDetection + MMCV` 环境上的。

## 3. 数据准备

`UnravelNet` 配置默认继承的是 `configs/_base_/datasets/dotav1.py`，其中默认数据根目录是：

```python
data_root = "/dataset/detection/DOTA10_mmrotate_split_ss/"
```

默认目录结构对应如下：

```text
/dataset/detection/DOTA10_mmrotate_split_ss/
├─ trainval/
│  ├─ annfiles/
│  └─ images/
├─ val/
│  ├─ annfiles/
│  └─ images/
└─ test/
   └─ images/
```

也就是说，开箱即用的配置默认是：

- 训练集：`trainval`
- 验证集：`val`
- 测试集：`test`

如果你的 DOTA 数据不在这个路径，最简单的做法有两个：

1. 直接改 `configs/_base_/datasets/dotav1.py` 里的 `data_root`
2. 启动训练/测试时用 `--cfg-options` 覆盖

例如，直接在 PowerShell 里写成单行：

```bash
python tools/train.py configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py --cfg-options data_root="D:/dataset/DOTA10_mmrotate_split_ss/"
```

## 4. 预训练权重

两个 `UnravelNet` 配置都默认加载骨干预训练权重：

- `./backbone_weights/unravelnet_tiny.pth`
- `./backbone_weights/unravelnet_small.pth`

但当前仓库根目录下还没有 `backbone_weights/` 文件夹，所以你需要自己补上。

README 里给了下载链接：

- `unravelnet_tiny.pth`
- `unravelnet_small.pth`

建议在仓库根目录新建：

```text
backbone_weights/
├─ unravelnet_tiny.pth
└─ unravelnet_small.pth
```

如果你暂时不想加载预训练，最稳妥的方式是直接修改对应配置文件里的：

```python
init_cfg=dict(type='Pretrained', checkpoint="./backbone_weights/unravelnet_small.pth")
```

把它改成：

```python
init_cfg=None
```

## 5. 训练

### 5.1 单卡训练

`UnravelNet-Tiny`：

```bash
python tools/train.py configs/unravelnet/ORCNN_unravelnet_tiny_fpn_le90_dota10val_ss_e36.py
```

`UnravelNet-Small`：

```bash
python tools/train.py configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py
```

指定输出目录：

```bash
python tools/train.py configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py --work-dir work_dirs/unravelnet_small_dota10
```

从断点继续训练：

```bash
python tools/train.py configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py --resume-from work_dirs/unravelnet_small_dota10/latest.pth
```

训练时不做验证：

```bash
python tools/train.py configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py --no-validate
```

### 5.2 多卡训练

Linux 下可以直接用：

```bash
bash tools/dist_train.sh configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py 4
```

Windows 下一般更稳妥的方式是先确认单卡流程，再用 PyTorch 分布式命令自行启动；仓库自带的 `dist_train.sh` 是 shell 脚本，主要面向 Linux。

## 6. 验证与测试

### 6.1 在验证集上算 mAP

```bash
python tools/test.py configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py work_dirs/unravelnet_small_dota10/latest.pth --eval mAP
```

### 6.2 可视化测试结果

```bash
python tools/test.py configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py work_dirs/unravelnet_small_dota10/latest.pth --show-dir work_dirs/unravelnet_small_vis
```

### 6.3 导出测试结果

```bash
python tools/test.py configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py work_dirs/unravelnet_small_dota10/latest.pth --out work_dirs/unravelnet_small.pkl
```

如果你要做 DOTA 提交格式导出，可以使用：

```bash
python tools/test.py configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py work_dirs/unravelnet_small_dota10/latest.pth --format-only --eval-options submission_dir=work_dirs/Task1_results
```

## 7. 单张图使用方法

最直接的是 `demo/image_demo.py`：

```bash
python demo/image_demo.py demo/demo.jpg configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py work_dirs/unravelnet_small_dota10/latest.pth --out-file work_dirs/demo_result.jpg --device cuda:0 --score-thr 0.3
```

作用是：

- 读取配置文件
- 加载 checkpoint
- 对单张图做检测
- 把可视化结果保存到 `--out-file`

如果想直接看窗口显示，可以不写 `--out-file`，但在很多远程环境里保存图片通常更稳。

## 8. 大图遥感影像使用方法

如果输入图像很大，可以用 `demo/huge_image_demo.py` 按 patch 切片推理：

```bash
python demo/huge_image_demo.py demo/dota_demo.jpg configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py work_dirs/unravelnet_small_dota10/latest.pth --patch_sizes 1024 --patch_steps 824 --img_ratios 1.0 --merge_iou_thr 0.1 --device cuda:0 --score-thr 0.3
```

这比较适合遥感大幅面图像，逻辑是：

- 先把大图裁成多个 patch
- 分别检测
- 再把结果按 IoU 阈值合并

## 9. 流程

1.  `Python / PyTorch / MMCV / MMDetection / MMRotate` 环境
2. 准备 DOTA 数据，并确认 `data_root` 正确
3. 下载 `unravelnet_tiny.pth` 或 `unravelnet_small.pth` 到 `backbone_weights/`
4. 先跑单卡训练
5. 用 `tools/test.py --eval mAP` 做验证
6. 用 `demo/image_demo.py` 或 `demo/huge_image_demo.py` 做可视化推理



- `configs/unravelnet/*`
- `mmrotate/models/backbones/unravelnet.py`
