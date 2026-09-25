# 基础镜像，选用轻量Python3.10
FROM python:3.10-slim
# 设置容器内工作目录
WORKDIR /app
# 把依赖文件复制进容器
COPY requirements.txt .

RUN pip install --upgrade pip
# 安装其余所有依赖（requirements已经移除torch）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=300 -r requirements.txt
# 单独安装cpu版 torch
RUN pip install --no-cache-dir --default-timeout=600 torch==2.13.0 --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cpu/

# 复制全部项目代码到容器
COPY . .
# 创建上传文件目录
RUN mkdir -p /app/uploads
RUN mkdir -p /app/logs
# 容器对外暴露端口8000（和代码uvicorn端口保持一致）
EXPOSE 8000
# 容器启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
