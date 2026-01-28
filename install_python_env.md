
# 基本信息

python==3.10
pytorch: 2.4.x
cuda: 12.4


# 需要单独安装的库
### 安装pytroch 
```bash
# 网址：https://pytorch.org/get-started/previous-versions/

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

### 安装xuance
```bash
# https://xuance.org/documents/usage/installation.html

pip install xuance[torch]

# cpu版本
pip install xuance
```

During the installation of XuanCe, you might encount the following error:

#Error: Failed to building wheel for mpi4py
#Failed to build mpi4py
#ERROR: Could not build wheels for mpi4py, which is required to install pyproject.toml-based projects
#Solution 1: You can solve that issue by install mpi4py manually via

```bash
conda install mpi4py
```


#Solution 2: If that doesn’t work, you can type and install gcc_linux-64 via:
```bash
conda install gcc_linux-64
#And then, retype the installation command for mpi4py via pip:

pip install mpi4py
```


### 安装dgl
```bash
# https://www.dgl.ai/pages/start.html

pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

```

### 其他
```bash
pip install -r requirements.txt
```


# 执行
```bash
python rl_reroute.py

```