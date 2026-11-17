# CIFAR-10 图像分类

用 PyTorch 实现一个包含三个卷积块的图像分类模型，对比训练时加入随机裁剪和水平翻转前后的表现。两组使用相同的模型、数据划分和训练设置，按验证准确率选择权重，再在 CIFAR-10 官方测试集上评估。

## 实验结果

| 训练设置 | 最佳轮次 | 验证准确率 | 测试准确率 |
| --- | ---: | ---: | ---: |
| A：无随机增强 | 20 | 75.76% | 76.01% |
| B：随机裁剪 + 水平翻转 | 19 | 70.18% | 71.09% |

在本次模型和 20 轮训练设置下，增强组的测试准确率低 4.92 个百分点。两组最常见的错分都发生在猫和狗之间，但变化方向不同：加入增强后，猫被判成狗从 143 张增至 204 张，狗被判成猫从 160 张减至 144 张。本次比较使用单个训练种子，且裁剪与翻转同时启用，未测量两种操作各自的影响。

## 模型与数据

- **模型**：三个卷积块，每个块依次使用 3×3 卷积、批归一化、ReLU 和 2×2 最大池化，通道数分别为 32、64、128。将最后的 128×4×4 特征展平后，接一个输出 10 个类别得分的全连接层。
- **数据**：[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 的 50,000 张训练图像按类别分层划分为 45,000 张训练和 5,000 张验证图像，划分种子为 2026。归一化统计量从训练子集计算，验证与测试均不做随机增强。
- **训练**：种子 42，批大小 128，SGD 初始学习率 0.05、动量 0.9、权重衰减 0.0005，使用余弦退火调度。B 组在训练时先以 4 像素填充做 32×32 随机裁剪，再以 0.5 的概率水平翻转。验证准确率相同时取较早轮次，权重锁定后使用官方 10,000 张测试图像评估。

## 运行

实验在 Windows、Python 3.11 和 PyTorch CPU 环境下运行。在仓库根目录打开 PowerShell：

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.15.0+cpu torchvision==0.29.2+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe experiment.py prepare --download
~~~

数据准备完成后，依次训练、绘图和评估：

~~~powershell
.\.venv\Scripts\python.exe experiment.py train --group A --name A_seed42 --epochs 20
.\.venv\Scripts\python.exe experiment.py train --group B --name B_seed42 --epochs 20
.\.venv\Scripts\python.exe experiment.py plot --runs runs/A_seed42 runs/B_seed42
.\.venv\Scripts\python.exe experiment.py lock --runs runs/A_seed42 runs/B_seed42
.\.venv\Scripts\python.exe experiment.py test --runs runs/A_seed42 runs/B_seed42
~~~

训练日志和权重写入各组的 `runs/` 子目录。测试输出包括总体及逐类准确率、混淆矩阵和错分图。
