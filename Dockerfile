# GT7 赛道工程师 —— 容器镜像
# ================================================================
# 🔴 纯标准库：**没有 requirements.txt，也不该有**。
#    整个包只 import 标准库（urllib / json / threading / http.server …），
#    所以镜像不需要 pip install 任何东西。加第三方依赖前先想清楚：
#    这个服务要在跑游戏的那台机器旁边常驻，依赖越少越不容易半夜挂。
FROM python:3.12-slim

WORKDIR /app

# 只拷包本体：tests / _ui_check 之类不进镜像
COPY gt7coach/ /app/gt7coach/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

# 8788 紧挨着仪表盘的 8787
EXPOSE 8788

# 默认：连宿主机上的 GT7 Dash（配合 network_mode: host）
ENTRYPOINT ["python", "-m", "gt7coach"]
CMD ["serve", "--dash", "http://127.0.0.1:8787", \
     "--bind", "0.0.0.0", "--port", "8788"]
