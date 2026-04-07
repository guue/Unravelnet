# UnravelNet DOTA1.0 Setup And Workflow

## 1. 额外建议安装的库

下面这些库我这次没有替你安装，只是给出建议。

### 建议安装

- `timm`
  - 用途：`UnravelNet`、`LEGNet`、`LSKNet` 这些 backbone 原始实现里会用到 `DropPath`、`trunc_normal_` 等工具。
  - 说明：我已经在代码里加了 fallback，让当前 `UnravelNet` 分支在没有 `timm` 时也能导入；但如果你想保持和作者原始依赖更一致，仍然建议安装。

### 按需安装

- `mmengine`
  - 用途：当前仓库中的 `PKINet` backbone 依赖它。
  - 说明：如果你不跑 `PKINet` 相关配置，可以暂时不装。

### 当前环境里已验证存在或可正常使用

- `torch`
- `torchvision`
- `mmcv-full`
- `mmdet`
- `opencv-python`
- `Pillow`
- `numpy`
- `shapely`
- `matplotlib`
- `antialiased-cnns`
- `e2cnn`

## 2. 本次我完成的处理

### 2.1 原始数据位置

你提供的原始 DOTA 1.0 数据位于：

- `dataset/DOTA10/train`
- `dataset/DOTA10/val`
- `dataset/DOTA10/test`

其中实际使用的是：

- 训练图像：`dataset/DOTA10/train/images`
- 验证图像：`dataset/DOTA10/val/images`
- 测试图像：`dataset/DOTA10/test/images`
- DOTA1.0 标注：
  - `dataset/DOTA10/train/labelTxt-v1.0/labelTxt`
  - `dataset/DOTA10/val/labelTxt-v1.0/labelTxt`

### 2.2 新增了本地专用的 DOTA1.0 切片配置

新增文件：

- [tools/data/dota/split/split_configs/local_dota10_ss_trainval.json](./tools/data/dota/split/split_configs/local_dota10_ss_trainval.json)
- [tools/data/dota/split/split_configs/local_dota10_ss_val.json](./tools/data/dota/split/split_configs/local_dota10_ss_val.json)
- [tools/data/dota/split/split_configs/local_dota10_ss_test.json](./tools/data/dota/split/split_configs/local_dota10_ss_test.json)

这些配置的含义是：

- patch 大小：`1024 x 1024`
- overlap：`200`
- 单尺度切片：`ss`
- 输出目录：
  - `dataset/DOTA10_mmrotate_split_ss/trainval`
  - `dataset/DOTA10_mmrotate_split_ss/val`
  - `dataset/DOTA10_mmrotate_split_ss/test`

### 2.3 修正了基础数据配置

修改文件：

- [configs/_base_/datasets/dotav1.py](./configs/_base_/datasets/dotav1.py)

修改内容：

- 将 `data_root` 改为：

```python
data_root = 'dataset/DOTA10_mmrotate_split_ss/'
```

这意味着后续只要使用基于 `dotav1.py` 的配置文件，默认就会读取你仓库里的本地切片数据。

### 2.4 生成了切片后的训练/验证/测试数据

我使用 `unravelnet` 环境里的 Python 跑了切片脚本。

实际使用的逻辑入口：

- [tools/data/dota/split/img_split.py](./tools/data/dota/split/img_split.py)

由于当前 Windows 环境下多进程切片会报权限错误，我对这个脚本做了一个最小兼容改动，使它在 `--nproc 1` 时走单进程路径。

### 2.5 切片结果

生成后的目录：

- `dataset/DOTA10_mmrotate_split_ss/trainval/images`
- `dataset/DOTA10_mmrotate_split_ss/trainval/annfiles`
- `dataset/DOTA10_mmrotate_split_ss/val/images`
- `dataset/DOTA10_mmrotate_split_ss/val/annfiles`
- `dataset/DOTA10_mmrotate_split_ss/test/images`

磁盘上的 patch 数量：

- `trainval`: `21046` 张 patch 图，`21046` 个标注文件
- `val`: `5297` 张 patch 图，`5297` 个标注文件
- `test`: `10833` 张 patch 图

### 2.6 导入兼容修复

为避免当前环境因为可选 backbone 或 `matplotlib` 缓存路径问题在导入阶段崩掉，我做了这些兼容修补：

- [mmrotate/models/backbones/__init__.py](./mmrotate/models/backbones/__init__.py)
  - 改为按需导入 `PKINet`、`LSKNet`、`LWEGNet`
- [mmrotate/models/backbones/unravelnet.py](./mmrotate/models/backbones/unravelnet.py)
  - 为 `timm` 缺失提供最小 fallback
- [mmrotate/__init__.py](./mmrotate/__init__.py)
  - 为 `matplotlib` 指定仓库内可写缓存目录 `.mplconfig`
- [sitecustomize.py](./sitecustomize.py)
  - 同样补了 `MPLCONFIGDIR` 的兜底设置

## 3. 我验证过的内容

### 3.1 已验证成功

- `mmrotate` 可以在当前 `unravelnet` 环境中导入。
- 切片目录真实存在。
- `DOTADataset` 可以直接读取这些切片目录。

我实际验证过的代码入口：

- [mmrotate/datasets/dota.py](./mmrotate/datasets/dota.py)

直接用 `DOTADataset` 读到的样本数：

- `trainval_len = 12800`
- `val_len = 3066`
- `test_len = 10833`

说明：

- 这里的运行时样本数和磁盘上的 patch 文件数不完全相同是正常的。
- 对训练/验证集，`DOTADataset` 会根据空标注、过滤规则等逻辑跳过一部分 patch。

### 3.2 还没有完全打通的部分

- 我没有在这台机器上完整跑通 `tools/train.py`。
- 原因不是数据路径缺失，而是当前环境里 `mmcv.Config.fromfile(...)` 在 Windows 中文路径场景下存在异常卡住现象，后续如果你要继续，我可以专门再帮你定位这一层。

也就是说：

- 数据已经配好
- 数据集类已经能读
- 训练入口脚本本身还有一层独立兼容问题待继续排查

## 4. 相关文档与关键代码

### 官方/仓库内相关文档

- [README.md](./README.md)
  - 仓库总体说明、推荐配置文件、预训练权重
- [tools/data/dota/README.md](./tools/data/dota/README.md)
  - DOTA 数据集准备与切片说明

### 本次直接相关的配置文件

- [configs/_base_/datasets/dotav1.py](./configs/_base_/datasets/dotav1.py)
  - DOTA1.0 数据根目录和 train/val/test 路径定义
- [configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py](./configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py)
  - 当前推荐的 UnravelNet DOTA1.0 单尺度训练配置
- [configs/_base_/schedules/schedule_3x_unravelnet.py](./configs/_base_/schedules/schedule_3x_unravelnet.py)
  - 优化器、学习率调度、训练轮数等

### 训练/验证/测试入口代码

- [tools/train.py](./tools/train.py)
  - 训练主入口
- [mmrotate/apis/train.py](./mmrotate/apis/train.py)
  - 实际构造 dataloader、runner、eval hook，并启动训练
- [tools/test.py](./tools/test.py)
  - 验证/测试/格式化提交主入口
- [mmrotate/datasets/dota.py](./mmrotate/datasets/dota.py)
  - DOTA patch 数据读取、评估、结果合并与提交格式化

### 分布式入口

- [tools/dist_train.sh](./tools/dist_train.sh)
- [tools/dist_test.sh](./tools/dist_test.sh)

## 5. 后续训练、验证、测试分别调用哪些代码

### 5.1 训练

你调用：

```powershell
C:\Users\22688\.conda\envs\unravelnet\python.exe tools\train.py configs\unravelnet\ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py
```

大致调用链：

1. `tools/train.py`
2. `Config.fromfile(...)` 读取配置
3. `build_detector(...)` 构造模型
4. `build_dataset(cfg.data.train)` 构造训练数据集
5. `mmrotate/apis/train.py::train_detector(...)`
6. `build_dataloader(...)`
7. `build_runner(...)`
8. `runner.run(...)`

训练时用到的数据：

- 训练集：`dataset/DOTA10_mmrotate_split_ss/trainval`
- 验证集：`dataset/DOTA10_mmrotate_split_ss/val`

### 5.2 训练过程中的验证

如果你不加 `--no-validate`，训练期间会自动做验证。

调用链：

1. `tools/train.py`
2. `mmrotate/apis/train.py`
3. `build_dataset(cfg.data.val, dict(test_mode=True))`
4. 注册 `EvalHook` / `DistEvalHook`
5. 按配置里的：

```python
evaluation = dict(interval=1, metric='mAP', save_best='mAP')
```

每个 epoch 跑一次验证并计算 `mAP`

### 5.3 单独验证

如果你已经有 checkpoint，通常会这样跑：

```powershell
C:\Users\22688\.conda\envs\unravelnet\python.exe tools\test.py ^
configs\unravelnet\ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py ^
work_dirs\ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36\latest.pth ^
--eval mAP
```

调用链：

1. `tools/test.py`
2. `Config.fromfile(...)`
3. `build_dataset(cfg.data.test)`
4. `build_dataloader(...)`
5. `build_detector(...)`
6. `load_checkpoint(...)`
7. `single_gpu_test(...)` 或 `multi_gpu_test(...)`
8. `dataset.evaluate(...)`

注意：

- `tools/test.py` 默认走 `cfg.data.test`
- 也就是当前配置里测试集对应的 `dataset/DOTA10_mmrotate_split_ss/test`
- 如果你想评估验证集 mAP，通常需要改配置或用 `--cfg-options` 把 `cfg.data.test` 临时改到 `cfg.data.val`

### 5.4 测试集预测/提交格式化

如果是 DOTA 测试集，没有真值，常见做法是：

```powershell
C:\Users\22688\.conda\envs\unravelnet\python.exe tools\test.py ^
configs\unravelnet\ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py ^
your_checkpoint.pth ^
--format-only
```

这里会用到：

- [mmrotate/datasets/dota.py](./mmrotate/datasets/dota.py)
  - `merge_det(...)`
  - `_results2submission(...)`

也就是先把 patch 级检测结果合并回原图，再组织成 DOTA 提交格式。

## 6. 整个工作流是什么样子

### 数据准备阶段

1. 下载 DOTA1.0 原始数据到 `dataset/DOTA10`
2. 使用 `img_split.py` 把原始大图切成 `1024x1024` patch，overlap `200`
3. 生成：
   - `trainval`
   - `val`
   - `test`
4. 修改 `configs/_base_/datasets/dotav1.py` 指向本地切片目录

### 训练阶段

1. 选择配置文件
   - 当前是 [configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py](./configs/unravelnet/ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py)
2. 加载预训练 backbone
   - `./backbone_weights/unravelnet_small.pth`
3. 构建训练集和验证集
4. 开始训练
5. 每个 epoch 自动做验证并保存最好结果

### 评估阶段

1. 载入训练好的 checkpoint
2. 在验证集上跑 `mAP`
3. 输出指标到终端或 `work_dir`

### 测试/提交阶段

1. 在切片后的 `test` patch 上跑推理
2. 将 patch 检测结果合并回原图
3. 生成 DOTA 测试提交结果

## 7. 你接下来最建议做的事情

### 如果你要继续推进

建议顺序是：

1. 先安装建议依赖：
   - `timm`
   - 如果之后要跑 `PKINet`，再装 `mmengine`
2. 尝试直接运行训练入口：

```powershell
C:\Users\22688\.conda\envs\unravelnet\python.exe tools\train.py configs\unravelnet\ORCNN_unravelnet_small_fpn_le90_dota10val_ss_e36.py
```

3. 如果还是在 `Config.fromfile(...)` 卡住，就继续定位这层 Windows/路径兼容问题

## 8. 本次修改过的文件

- [configs/_base_/datasets/dotav1.py](./configs/_base_/datasets/dotav1.py)
- [tools/data/dota/split/split_configs/local_dota10_ss_trainval.json](./tools/data/dota/split/split_configs/local_dota10_ss_trainval.json)
- [tools/data/dota/split/split_configs/local_dota10_ss_val.json](./tools/data/dota/split/split_configs/local_dota10_ss_val.json)
- [tools/data/dota/split/split_configs/local_dota10_ss_test.json](./tools/data/dota/split/split_configs/local_dota10_ss_test.json)
- [tools/data/dota/split/img_split.py](./tools/data/dota/split/img_split.py)
- [mmrotate/models/backbones/__init__.py](./mmrotate/models/backbones/__init__.py)
- [mmrotate/models/backbones/unravelnet.py](./mmrotate/models/backbones/unravelnet.py)
- [mmrotate/__init__.py](./mmrotate/__init__.py)
- [sitecustomize.py](./sitecustomize.py)
