import sys
import os
import subprocess
import time
import argparse
import json
import socket

# Find workspace directory dynamically (go up until we find frontend folder)
current = os.path.abspath(os.path.dirname(__file__))
while current and not os.path.exists(os.path.join(current, "frontend")) and os.path.dirname(current) != current:
    current = os.path.dirname(current)
WORKSPACE_DIR = current
os.chdir(WORKSPACE_DIR)

def check_port(port):
    """Check if a port is open/listening on localhost (IPv4 or IPv6)."""
    for host in ["127.0.0.1", "::1"]:
        try:
            # Determine family
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect((host, port))
                return True
        except Exception:
            continue
    return False

def get_pids_by_port(port):
    """Find all PIDs listening on a specific port using netstat on Windows."""
    pids = set()
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                local_addr = parts[1]
                if local_addr.endswith(f":{port}"):
                    pid = parts[-1]
                    if pid.isdigit() and pid != "0":
                        pids.add(int(pid))
    except Exception as e:
        print(f"[WARN] Error running netstat: {e}", file=sys.stderr)
    return list(pids)

def kill_process_by_port(port):
    """Kill all processes listening on a specific port."""
    pids = get_pids_by_port(port)
    if not pids:
        print(f"No active process found listening on port {port}.")
        return True
    
    success = True
    for pid in pids:
        try:
            print(f"Terminating process PID {pid} listening on port {port}...")
            # /F forces termination, /T terminates child processes too
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], check=True)
            print(f"Successfully terminated process {pid}.")
        except Exception as e:
            print(f"[ERROR] Failed to terminate process {pid}: {e}", file=sys.stderr)
            success = False
    return success

def check_docker_status():
    """Check if the spotsync-postgis container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=spotsync-postgis", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            check=True
        )
        status = result.stdout.strip()
        if "Up" in status:
            return "Running"
        elif status:
            return f"Stopped ({status})"
        else:
            return "Not Created"
    except Exception:
        return "Docker daemon not running / Unreachable"

def start_services(args):
    """Start Docker, Backend API, and Frontend dev servers."""
    print("=== Starting Gravity Services ===")
    os.makedirs("logs", exist_ok=True)

    # 1. Start Docker container
    print("1. Starting PostgreSQL PostGIS DB container...")
    try:
        subprocess.run(["docker-compose", "up", "-d", "postgis"], check=True)
        print("   Docker container started successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to start Docker container: {e}", file=sys.stderr)
        print("   Make sure Docker Desktop is running.", file=sys.stderr)

    # 2. Start Backend API Server
    if check_port(8000):
        print("2. [WARN] Port 8000 is already in use. Skipping backend startup.")
    else:
        print("2. Starting FastAPI Backend on port 8000...")
        python_bin = os.path.abspath(".venv/Scripts/python.exe") if os.path.exists(".venv/Scripts/python.exe") else "python"
        backend_log_path = os.path.abspath("logs/backend.log")
        backend_err_path = os.path.abspath("logs/backend.err")
        
        # Clear existing logs
        for path in [backend_log_path, backend_err_path]:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            
        cmd = [
            "powershell", "-Command",
            f"Start-Process -FilePath '{python_bin}' -ArgumentList '-u', '-m', 'uvicorn', 'ai_search_v10:app', '--host', '0.0.0.0', '--port', '8000' -RedirectStandardOutput '{backend_log_path}' -RedirectStandardError '{backend_err_path}' -WindowStyle Hidden"
        ]
        subprocess.run(cmd, check=True)
        print("   Backend server spawned in the background (log: logs/backend.log).")

    # 3. Start Frontend Dev Server
    if check_port(5173):
        print("3. [WARN] Port 5173 is already in use. Skipping frontend startup.")
    else:
        print("3. Starting Vite Frontend on port 5173...")
        frontend_log_path = os.path.abspath("logs/frontend.log")
        frontend_err_path = os.path.abspath("logs/frontend.err")
        frontend_dir = os.path.abspath("frontend")
        
        # Clear existing logs
        for path in [frontend_log_path, frontend_err_path]:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            
        cmd = [
            "powershell", "-Command",
            f"Start-Process -FilePath 'npm.cmd' -ArgumentList 'run', 'dev' -WorkingDirectory '{frontend_dir}' -RedirectStandardOutput '{frontend_log_path}' -RedirectStandardError '{frontend_err_path}' -WindowStyle Hidden"
        ]
        subprocess.run(cmd, check=True)
        print("   Frontend dev server spawned in the background (log: logs/frontend.log).")

    # 4. Wait and verify
    print("\nVerifying service startup (waiting 15 seconds)...")
    time.sleep(15)
    print_status()

def stop_services(args):
    """Stop Backend API, Frontend dev servers, and Docker container."""
    print("=== Stopping Gravity Services ===")
    
    # 1. Stop Frontend
    print("1. Stopping Vite Frontend on port 5173...")
    kill_process_by_port(5173)

    # 2. Stop Backend
    print("2. Stopping FastAPI Backend on port 8000...")
    kill_process_by_port(8000)

    # 3. Stop Docker DB
    print("3. Stopping PostgreSQL PostGIS DB container...")
    try:
        subprocess.run(["docker-compose", "down"], check=True)
        print("   Docker container stopped successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to stop Docker container: {e}", file=sys.stderr)

    print("\n=== Services Stopped successfully ===")

def print_status():
    """Print the current running status of all services."""
    docker_status = check_docker_status()
    backend_status = "Running" if check_port(8000) else "Stopped"
    frontend_status = "Running" if check_port(5173) else "Stopped"

    print("=== Gravity Services Status ===")
    print(f"  PostgreSQL PostGIS (Docker): {docker_status}")
    print(f"  FastAPI Backend (Port 8000): {backend_status}")
    print(f"  Vite Frontend   (Port 5173): {frontend_status}")
    
    # If running, print PIDs
    if backend_status == "Running":
        print(f"    Backend PIDs: {get_pids_by_port(8000)}")
    if frontend_status == "Running":
        print(f"    Frontend PIDs: {get_pids_by_port(5173)}")
    print("===============================")

def status_cmd(args):
    print_status()

def main():
    parser = argparse.ArgumentParser(description="Gravity Service Management Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("start", help="Start all services (Docker, Backend, Frontend)")
    subparsers.add_parser("stop", help="Stop all services")
    subparsers.add_parser("status", help="Get current status of all services")

    args = parser.parse_args()

    if args.command == "start":
        start_services(args)
    elif args.command == "stop":
        stop_services(args)
    elif args.command == "status":
        status_cmd(args)

if __name__ == "__main__":
    main()
