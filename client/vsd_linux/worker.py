import argparse
import asyncio
import binascii
import os
import re
import subprocess
import time

import aiohttp

IP = "http://192.168.0.10:8000/echo"
LAST_EVENT_TIME = time.time()


async def send_data(data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(IP, json=data) as response:
                if response.status == 200:
                    print("Data sent successfully")
                else:
                    print(f"Failed to send data: {response.status}")
    except Exception as e:
        print(f"Error sending data: {e}")


async def monitor(parent_pid: int):
    monitor_pid = os.getpid()
    await send_data({"pid": monitor_pid, "parent_pid": parent_pid, "time": LAST_EVENT_TIME})
    while True:
        await asyncio.sleep(3)
        try:
            process = subprocess.run(
                ['sudo', '/sbin/ausearch', '-k', 'bash_activity'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.returncode == 0:
                output = process.stdout.decode()
                output = parse_audit_log_blocks(output)
                if len(output) == 0:
                    continue
                output = filter_old_entries(output)
                output = filter_entries_by_command(output, ['ausearch'])
                output = filter_entries_by_ppid(output, [parent_pid, monitor_pid])
                if len(output) == 0:
                    continue
                await send_data({"output": output})
            else:
                error = process.stderr.decode()
                await send_data({"warning": error})
        except Exception as e:
            await send_data({"error": str(e)})


def parse_audit_log_blocks(log_data: str):
    entries = []
    blocks = [b.strip() for b in log_data.strip().split('----') if b.strip()]

    for block in blocks:
        entry = {
            "timestamp": None,
            "success": None,
            "uid": None,
            "euid": None,
            "syscall": None,
            "ppid": None,
            "pid": None,
            "command": None,
            "arguments": [],
            "CWD": None
        }

        for line in block.splitlines():
            # Extract timestamp
            if line.startswith("time->"):
                continue  # skip human-readable time

            if match := re.search(r'audit\(([\d.]+):\d+\)', line):
                entry["timestamp"] = float(match.group(1).split(':')[0])

            if 'type=SYSCALL' in line:
                fields = dict(re.findall(r'(\w+)=("[^"]+"|\S+)', line))
                entry["success"] = 1 if fields.get("success") == "yes" else 0
                entry["uid"] = int(fields.get("uid"))
                entry["euid"] = int(fields.get("euid"))
                entry["syscall"] = int(fields.get("syscall"))
                entry["ppid"] = int(fields.get("ppid"))
                entry["pid"] = int(fields.get("pid"))
                entry["command"] = fields.get("comm", "").strip('"')

            elif 'type=PROCTITLE' in line:
                if match := re.search(r'proctitle=([0-9A-Fa-f]+)', line):
                    hex_str = match.group(1)
                    try:
                        decoded = binascii.unhexlify(hex_str).decode(errors='replace')
                        entry["arguments"] = decoded.split('\x00')
                    except Exception:
                        entry["arguments"] = []

            elif 'type=CWD' in line:
                if match := re.search(r'cwd="([^"]+)"', line):
                    entry["CWD"] = match.group(1)

        if entry["timestamp"] is not None:
            entries.append(entry)

    return entries


def filter_old_entries(entries):
    global LAST_EVENT_TIME
    filtered_entries = [i for i in entries if i["timestamp"] > LAST_EVENT_TIME]
    LAST_EVENT_TIME = filtered_entries[-1]["timestamp"]
    return filtered_entries


def filter_entries_by_command(entries: list, excluded_commands: list[str]) -> list:
    if len(entries) == 0:
        return entries
    return [
        entry for entry in entries
        if entry["command"] not in excluded_commands
    ]


def filter_entries_by_ppid(entries: list, excluded_pids: list[int] | int) -> list:
    if len(entries) == 0:
        return entries
    if isinstance(excluded_pids, int):
        excluded_pids = [excluded_pids]
    return [
        entry for entry in entries
        if entry["ppid"] not in excluded_pids
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='vsd-linux-worker')
    parser.add_argument("--parent-pid", type=int, help="Parent process ID to exclude from monitoring")
    args = parser.parse_args()

    asyncio.run(monitor(parent_pid=args.parent_pid))
