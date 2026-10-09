# -*- coding: utf-8 -*-
"""部署到远程服务器：上传 → md5 硬校验 → 重建镜像 → 重启容器。

用法：
    set GT7_SERVER_PASSWORD=...            # Windows
    export GT7_SERVER_PASSWORD=...         # Linux/macOS
    python tools/deploy.py --host 192.168.43.18

🔴 凭据只从环境变量（或 --password-file）读，**绝不写进仓库**。
   用户名默认 root，也可用 GT7_SERVER_USER 覆盖。

🔴 为什么必须 md5 硬校验：镜像里的代码靠 Dockerfile 的 COPY 烤进去，
   "文件传到服务器了" ≠ "镜像里是这份"。改完代码忘了重建镜像时，
   磁盘上的文件是新的、容器里跑的还是旧的，而现象只是"新功能没生效" ——
   最容易误判成"代码写错了"。

跑在能同时看到两个仓库的机器上（--dash-dir 指向 gt7-dash 的检出目录）。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

try:
    import paramiko
except ImportError:                       # pragma: no cover
    sys.exit("需要 paramiko：pip install paramiko")

DASH_FILES = ["gt7-dashboard.py", "gt7analysis.py", "gt7-recorder.py"]
COACH_TOP = ["Dockerfile", "docker-compose.yml", ".dockerignore"]


def md5_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


class Deployer:
    def __init__(self, host: str, user: str, password: str):
        self.cli = paramiko.SSHClient()
        self.cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.cli.connect(host, username=user, password=password, timeout=15)
        self.sftp = self.cli.open_sftp()

    def run(self, cmd: str, timeout: int = 900, quiet: bool = False) -> str:
        _i, out, err = self.cli.exec_command(cmd, timeout=timeout)
        o = out.read().decode("utf-8", "replace")
        e = err.read().decode("utf-8", "replace")
        text = (o + e).strip()
        if not quiet and text:
            print(text[-1800:])
        return text

    def put_verified(self, local: str, remote: str) -> None:
        """上传 + md5 比对，不一致直接抛。"""
        self.sftp.put(local, remote)
        remote_md5 = self.run(f"md5sum {remote} | cut -d' ' -f1", quiet=True)
        local_md5 = md5_file(local)
        ok = local_md5 == remote_md5
        print(f"    {os.path.basename(remote):<24} {local_md5[:12]}  "
              f"{'✓' if ok else '✗ ' + remote_md5[:12]}")
        if not ok:
            raise SystemExit("md5 不一致，中止（别在没确认的前提下重建镜像）")

    def container_md5(self, container: str, path: str) -> str:
        return self.run(f"docker exec {container} md5sum {path} 2>/dev/null "
                        f"| cut -d' ' -f1", quiet=True)

    def close(self) -> None:
        self.sftp.close()
        self.cli.close()


def deploy_dash(d: Deployer, dash_dir: str, skip: bool) -> None:
    if skip:
        print("\n[1/3] 跳过 GT7 Dash（--skip-dash）")
        return
    print("\n[1/3] GT7 Dash：备份 → 上传 → md5 校验 → 重建镜像 → 重启")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    d.run(f"mkdir -p /opt/gt7-recorder/_backup/{stamp} && cp -a "
          f"/opt/gt7-recorder/{' /opt/gt7-recorder/'.join(DASH_FILES)} "
          f"/opt/gt7-recorder/_backup/{stamp}/", quiet=True)
    print(f"    已备份到 /opt/gt7-recorder/_backup/{stamp}/")
    for name in DASH_FILES:
        d.put_verified(os.path.join(dash_dir, name),
                       f"/opt/gt7-recorder/{name}")
    print("    重建镜像…")
    d.run("cd /opt/gt7-recorder && docker build -t gt7-recorder:latest . "
          "2>&1 | tail -6")
    # ⚠️ 只重启这两个服务，别 up -d 整个项目（会连带动到别的容器）
    d.run("cd /opt/gt7-recorder && docker compose up -d --force-recreate "
          "--no-deps gt7-recorder gt7-dashboard 2>&1 | tail -6")


def deploy_coach(d: Deployer, coach_dir: str) -> None:
    print("\n[2/3] GT7 Coach：上传 → 重建镜像 → 重启")
    d.run("mkdir -p /opt/gt7-coach/gt7coach", quiet=True)
    for name in COACH_TOP:
        d.put_verified(os.path.join(coach_dir, name), f"/opt/gt7-coach/{name}")
    pkg = os.path.join(coach_dir, "gt7coach")
    for fn in sorted(os.listdir(pkg)):
        if fn.endswith(".py"):
            d.put_verified(os.path.join(pkg, fn),
                           f"/opt/gt7-coach/gt7coach/{fn}")
    d.run("cd /opt/gt7-coach && docker compose up -d --build 2>&1 | tail -6")


def verify(d: Deployer, dash_dir: str, coach_dir: str,
           skip_dash: bool = False) -> list[str]:
    """硬门槛：容器里的代码 md5 == 本地。

    `skip_dash=True` 时跳过 dash 那三个文件 —— 否则会在 `--dash-dir` 没给
    的时候去读 `./gt7-dashboard.py` 直接 FileNotFoundError 崩掉
    （**这次就是这样**：部署其实成功了、体检也全绿，最后一步自己炸了）。
    """
    print("\n[3/3] 硬校验：容器内代码 == 本地")
    bad = []
    if skip_dash:
        print("    （--skip-dash：跳过 Dash 三个文件）")
    for name in ([] if skip_dash else DASH_FILES):
        lm = md5_file(os.path.join(dash_dir, name))
        rm = d.container_md5("gt7-dashboard", f"/app/{name}")
        ok = lm == rm
        print(f"    dash/{name:<22} {'✓' if ok else '✗ ' + rm[:12]}")
        if not ok:
            bad.append(name)
    pkg = os.path.join(coach_dir, "gt7coach")
    for fn in sorted(os.listdir(pkg)):
        if not fn.endswith(".py"):
            continue
        lm = md5_file(os.path.join(pkg, fn))
        rm = d.container_md5("gt7-coach", f"/app/gt7coach/{fn}")
        if lm != rm:
            print(f"    coach/{fn:<22} ✗")
            bad.append(fn)
    if not bad:
        print("    coach 全部 .py 一致 ✓")
    print("\n" + d.run("docker ps --filter name=gt7 --format "
                       "'{{.Names}}\t{{.Status}}'"))
    return bad


def main(argv=None) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="部署 GT7 Dash + Coach 到服务器")
    ap.add_argument("--host", default=os.environ.get("GT7_SERVER_HOST", ""))
    ap.add_argument("--user", default=os.environ.get("GT7_SERVER_USER", "root"))
    ap.add_argument("--password", default=os.environ.get("GT7_SERVER_PASSWORD", ""))
    ap.add_argument("--password-file", default="",
                    help="从文件读密码（文件应 gitignore）")
    ap.add_argument("--coach-dir", default=here,
                    help="gt7-coach 检出目录，缺省为本脚本所在仓库")
    ap.add_argument("--dash-dir", default=os.environ.get("GT7_DASH_DIR", ""),
                    help="gt7-dash 检出目录")
    ap.add_argument("--skip-dash", action="store_true")
    args = ap.parse_args(argv)

    pw = args.password
    if not pw and args.password_file:
        with open(args.password_file, encoding="utf-8") as fh:
            pw = fh.read().strip()
    if not args.host or not pw:
        ap.error("需要 --host 和密码（--password / GT7_SERVER_PASSWORD / "
                 "--password-file 三者之一）")
    if not args.skip_dash and not args.dash_dir:
        ap.error("要部署 Dash 请给 --dash-dir；只想更新 Coach 用 --skip-dash")

    d = Deployer(args.host, args.user, pw)
    try:
        deploy_dash(d, args.dash_dir, args.skip_dash)
        deploy_coach(d, args.coach_dir)
        bad = verify(d, args.dash_dir or ".", args.coach_dir,
                     skip_dash=args.skip_dash)
    finally:
        d.close()
    if bad:
        print(f"\n❌ 以下文件容器内与本地不一致：{bad}")
        return 1
    print("\n✅ 部署完成，容器内代码与本地逐字节一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
