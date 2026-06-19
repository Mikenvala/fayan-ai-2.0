# 法眼AI 2.0 Docker 镜像
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY internship-tasks/task4-unified-platform/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 复制全部项目文件
COPY . .

# 设置工作目录
WORKDIR /app/internship-tasks/task4-unified-platform

# 暴露端口
EXPOSE 8800

# 启动脚本：先生成数据再启动服务
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
