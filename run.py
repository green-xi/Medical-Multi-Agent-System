import os, sys, time, subprocess, urllib.request, urllib.error
from threading import Thread

VENV_DIR = ".venv"
IS_WINDOWS = sys.platform == "win32"
PYTHON_EXE = os.path.join(VENV_DIR, "Scripts", "python.exe") if IS_WINDOWS else os.path.join(VENV_DIR, "bin", "python")
BACKEND_PORT = int(os.environ.get("MEDICALAI_BACKEND_PORT", "8000"))
BACKEND_URL = f"http://localhost:{BACKEND_PORT}"
FORCE_RESTART_BACKEND = (
    "--restart-backend" in sys.argv or
    os.environ.get("MEDICALAI_RESTART_BACKEND", "0") == "1"
)

def setup():
    # Create virtual environment if it doesn't exist
    if not os.path.exists(VENV_DIR):
        print(f"正在创建虚拟环境：{VENV_DIR}...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)

    # Auto-install dependencies
    if os.path.exists("backend/requirements.txt"):
        print("检查后端依赖…")
        # Use the venv's python to install requirements
        subprocess.run([PYTHON_EXE, "-m", "pip", "install", "-r", "backend/requirements.txt"])
    
    if not os.path.exists("frontend/node_modules"):
        print("安装前端依赖…")
        subprocess.run(["npm", "install"], cwd="frontend", shell=True)

def wait_for_api():
    print(f"等待后端启动，端口 {BACKEND_PORT}（首次启动需加载模型和向量库）…")
    for _ in range(300):
        try:
            if urllib.request.urlopen(f"{BACKEND_URL}/api/v1/health").status == 200:
                print("后端就绪！")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("后端未能在规定时间内就绪。")
    return False


def port_is_healthy():
    try:
        with urllib.request.urlopen(f"{BACKEND_URL}/api/v1/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def port_is_busy():
    try:
        with urllib.request.urlopen(BACKEND_URL, timeout=2):
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def find_port_pids(port: int):
    if IS_WINDOWS:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = set()
        needle = f":{port}"
        for line in result.stdout.splitlines():
            if needle not in line or "LISTENING" not in line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                pids.add(parts[-1])
        return sorted(pids)

    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted({pid.strip() for pid in result.stdout.splitlines() if pid.strip()})


def stop_port_processes(port: int):
    pids = find_port_pids(port)
    if not pids:
        return False

    print(f"正在停止占用端口 {port} 的进程：{', '.join(pids)}")
    for pid in pids:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", pid, "/F"], check=False)
        else:
            subprocess.run(["kill", "-9", pid], check=False)

    for _ in range(20):
        if not port_is_busy():
            return True
        time.sleep(0.5)
    return False

if __name__ == "__main__":
    setup()

    if FORCE_RESTART_BACKEND and port_is_busy():
        if not stop_port_processes(BACKEND_PORT):
            print(f"无法释放后端端口 {BACKEND_PORT}，请手动终止占用进程。")
            sys.exit(1)

    backend_running = port_is_healthy()
    if backend_running:
        print(f"复用已运行的后端：{BACKEND_URL}")
    elif port_is_busy():
        print(
            f"端口 {BACKEND_PORT} 已被其他进程占用，"
            "且该进程不是 MedicalAI 后端服务。"
        )
        sys.exit(1)
    else:
        # Start backend in background using venv's python
        env = {**os.environ, "PYTHONPATH": os.path.abspath("backend")}
        Thread(target=lambda: subprocess.run(
            [PYTHON_EXE, "-m", "uvicorn", "app.main:app", "--port", str(BACKEND_PORT)],
            cwd="backend", env=env
        ), daemon=True).start()

    # Wait and launch frontend
    if wait_for_api():
        subprocess.run(["npm", "run", "dev"], cwd="frontend", shell=True)
