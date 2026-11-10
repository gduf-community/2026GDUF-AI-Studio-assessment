FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 默认走官方源；国内网络可用 --build-arg 切换镜像，例如：
#   --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#   --build-arg "PIP_EXTRA=--trusted-host pypi.tuna.tsinghua.edu.cn"
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA=
RUN pip install --no-cache-dir -i "${PIP_INDEX_URL}" ${PIP_EXTRA} -r requirements.txt

COPY src ./src
COPY run.sh train.sh test.sh ./
RUN chmod +x ./*.sh

CMD ["/bin/bash", "/app/run.sh"]
