import getpass
import os
import re
import subprocess
from pathlib import Path

import psutil
import requests

PID_FILE = Path('/tmp/vsd_monitor.pid')
ENDPOINT = 'http://your.endpoint/receive'
HZ = Path('/home/ezka/bashiwashie/output.txt')
IP = "http://192.168.0.10:8000/echo"


def start_monitor():
    parent_pid = os.getppid()

    if not status_monitor():
        start_monitor_service()
        if not status_monitor():
            raise Exception("Failed to start auditd service.")

    setup_audit()
    setup_passwordless_sudo()
    print("Audit rules and passwordless sudo configured.")

    if is_process_running():
        print("Monitor is already running.")
        return

    # Detach worker.py and redirect output to log
    with open('monitor_stdout.log', 'a') as out, open('monitor_stderr.log', 'a') as err:
        process = subprocess.Popen(
            [os.path.join(os.getcwd(), '.venv/bin/python'), 'vsd_linux/worker.py', '--parent-pid', str(parent_pid)],
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            start_new_session=True
        )
        # Save the PID for management
        with open(PID_FILE, 'w') as f:
            f.write(str(process.pid))
        print(f"Monitor started with PID {process.pid}")


def status_monitor() -> bool:
    try:
        process = subprocess.run(
            ['sudo', 'systemctl', 'status', 'auditd'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        try:
            output = process.stdout.decode()
            status_match = re.search(r'^\s*Active:\s+(.+)', output, re.MULTILINE)
            if status_match:
                status = status_match.group(1).split()[0]
                if status == 'active':
                    return True
                else:
                    return False
            else:
                return False
        except Exception as parse_error:
            print(f"Error parsing status output: {parse_error}")
            return False
    except subprocess.CalledProcessError as cmd_error:
        print(f"Command failed with exit code {cmd_error.returncode}")
        print(f"stderr: {cmd_error.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def start_monitor_service() -> None:
    try:
        _ = subprocess.run(
            ['sudo', 'systemctl', 'start', 'auditd'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr.decode()}")
        else:
            print("No stderr captured.")
    except Exception as e:
        print(f"Unexpected error: {e}")


def setup_audit() -> None:
    rules_path = '/etc/audit/rules.d/bash_activity.rules'
    rules = (
        '-a always,exit -F arch=b64 -S execve -F auid>=1000 -F auid!=-1 -F key=bash_activity\n'
        '-a always,exit -F arch=b32 -S execve -F auid>=1000 -F auid!=-1 -F key=bash_activity\n')

    try:
        process = subprocess.run(
            ['sudo', 'tee', rules_path],
            input=rules.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        process = subprocess.run(
            ['sudo', 'auditctl', '-D'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )  ## Flush existing rules
        process = subprocess.run(
            ['sudo', 'augenrules', '--load'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr.decode()}")
        else:
            print("No stderr captured.")
    except Exception as e:
        print(f"Unexpected error: {e}")


def setup_passwordless_sudo(user: str = None):
    command_path = subprocess.run(['which', 'ausearch'], stdout=subprocess.PIPE).stdout.decode().strip()
    if os.geteuid() != 0:
        raise PermissionError("This script must be run as root to modify sudoers.")

    if user is None:
        user = getpass.getuser()

    sudoers_line = f"{user} ALL=(ALL) NOPASSWD: {command_path}\n"
    sudoers_file = Path(f"/etc/sudoers.d/ausearch_nopasswd")

    if sudoers_file.exists():
        with sudoers_file.open("r") as f:
            if sudoers_line in f.read():
                print("Passwordless sudo already configured.")
                return

    with sudoers_file.open("w") as f:
        f.write(sudoers_line)
        print(f"Added passwordless sudo rule for {command_path}")

    # Set correct permissions
    os.chmod(sudoers_file, 0o440)


def is_process_running() -> bool:
    if PID_FILE.exists():
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
            return psutil.pid_exists(pid)
    return False


def stop_monitor() -> None:
    if is_process_running():
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
            os.kill(pid, 9)
            print(f"Monitor stopped with PID {pid}")
        PID_FILE.unlink(missing_ok=True)
    else:
        print("Monitor is not running.")


def is_process_running() -> bool:
    # Dummy implementation — replace with your actual check
    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except:
        return False


def status() -> None:
    endpoint_status = "Unknown"

    # Check if endpoint is reachable
    try:
        response = requests.post(IP, json={"ping": "test"}, timeout=2)
        if response.status_code == 200:
            endpoint_status = "Connected"
        else:
            endpoint_status = f"Unreachable (HTTP {response.status_code})"
    except requests.RequestException as e:
        endpoint_status = f"Unreachable ({e.__class__.__name__})"

    # Check if monitor is running
    if is_process_running():
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
            print(f"Monitor is running with PID {pid}")
    else:
        print("Monitor is not running.")

    print(f"Endpoint status: {endpoint_status}")
