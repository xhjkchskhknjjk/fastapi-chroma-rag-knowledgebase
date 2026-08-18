# 基础镜像，选用轻量Python3.10
FROM python:3.10-slim

# 设置容器内工作目录
WORKDIR /app

# 把依赖文件复制进容器
COPY requirements.txt .

# 安装依赖，关闭缓存减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 复制全部项目代码到容器
COPY . .

# 创建上传文件目录
RUN mkdir -p /app/uploads

# 容器对外暴露端口8000（和代码uvicorn端口保持一致）
EXPOSE 8000

# 容器启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]