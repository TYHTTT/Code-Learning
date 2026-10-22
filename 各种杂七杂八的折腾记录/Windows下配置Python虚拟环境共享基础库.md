# 公用基础库

创建公共 PyTorch 基础环境：

```powershell
python -m venv D:/PythonVirtualEnvironment/PyTorchBasicVirtualEnvironment
```

激活虚拟环境：

```powershell
D:#进入D盘
cd D:/PythonVirtualEnvironment/PyTorchBasicVirtualEnvironment
./Scripts/Activate.ps1
```

安装 PyTorch(以 2.13.0+CUDA 13.0为例)：

```powershell
#临时设置 powershell 系统代理
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"

pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130

#验证 PyTorch 是否安装成功
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

# 项目虚拟环境

进入项目根目录创建项目虚拟环境并激活(以原版 3DGS 为例)：

```powershell
D:
cd D:/.Coding-Projects/gaussian-splatting
python -m venv .venv
./.venv/Scripts/Activate.ps1
```

在已激活的项目虚拟环境中设置：

```powershell
#连接到公用基础虚拟环境
$env:PYTHONPATH = "D:/PythonVirtualEnvironment/PyTorchBasicVirtualEnvironment/Lib/site-packages"

#验证是否连接到公用基础虚拟环境
python -c "import torch; print(torch.__version__); print(torch.__file__); print(torch.cuda.is_available())"

#安装需要本地编译的包
pip install ./submodules/diff-gaussian-rasterization --no-build-isolation --no-deps
pip install ./submodules/simple-knn --no-build-isolation --no-deps
pip install ./submodules/fused-ssim --no-build-isolation --no-deps

#安装后验证
python -c "import diff_gaussian_rasterization; import simple_knn._C; import fused_ssim; print('All CUDA extensions OK')"
```

下载官方场景数据：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip

解压到项目根目录。

运行代码，例如：

```powershell
python train.py -s tandt_db/tandt/train
```

