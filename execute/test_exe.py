import subprocess
import json

# 定义命令和参数
cmd = ["python", "test.py", "--src", "A", "--dst", "E", "--bw", "10"]

try:
    # 1. 运行命令
    # capture_output=True: 捕获输出
    # text=True: 将输出作为字符串处理（而不是字节）
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # 2. 获取 stdout (标准输出) 中的字符串
    json_str = result.stdout
    print(f"收到原始字符串: {json_str.strip()}")

    # 3. 解析 JSON 为字典
    path_data = json.loads(json_str)

    # 4. 使用数据
    if path_data["success"]:
        print(f"路径计算成功，跳数为: {len(path_data['path'])}")
    else:
        print("路径计算失败")

except subprocess.CalledProcessError as e:
    print(f"脚本运行出错: {e.stderr}")
except json.JSONDecodeError:
    print("脚本输出的不是合法的 JSON")