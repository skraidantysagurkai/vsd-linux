import subprocess
import re
import traceback
import time
import pandas as pd

from mongo.db_script import ClientMongo
from typing import List, Dict, Union, Optional
from datetime import datetime, timedelta


def run_ausearch() -> Optional[str]:
    """
    Run the ausearch command to get the logs.
    """
    result = subprocess.run(['sudo','ausearch', '-k', 'bash_activity'],
                            capture_output=True,
                            text=True)
    
    if result.returncode != 0:
        print(f"Error running ausearch: {result.stderr}")
        return None
    
    return result.stdout


def parse_log(log:str) -> Optional[Dict[str, Union[str, int]]]:
    """
    Parse a single auditd log entry into structured data.
    """
    lines = [line for line in log.split('\n') if line != '']
    
    if not lines:
        print("Empty log entry.")
        return
    
    if len(lines) != 8:
        return
    
    # timestamp
    try:
        timestamp = lines[0].split('->')[-1]
        date_obj = datetime.strptime(timestamp, "%a %b %d %H:%M:%S %Y")
        # skip old logs
        if datetime.utcnow() - date_obj > timedelta(minutes=5):
            return
        timestamp = int(date_obj.timestamp())
    except Exception as e:
        print(f"Error parsing timestamp: {e}")
        traceback.print_exc()
        timestamp = ''
        
    # curent working directory
    try:
        cwd_match = re.search(r'cwd="([^"]+)"', '\n'.join(lines))
        cwd = cwd_match.group(1) if cwd_match else ''
    except Exception as e:
        print(f"Error parsing cwd: {e}")
        traceback.print_exc()
        cwd = ''
    
    # argument count
    try:
        execve_line = lines[6]
        arg_count = int(re.search(r'argc=(\d+)', execve_line).group(1))
    except Exception as e:
        print(f"Error parsing arg count: {e}")
        traceback.print_exc()
        arg_count = 0

    # command
    if arg_count != 0:
        try:
            commands = []
            for arg in range(arg_count):
                commands.append(re.search(f'a{arg}="?([^"\s;]+)"?', execve_line).group(1))
            command = ' '.join(commands)
        except Exception as e:
            print(f"Error parsing command: {e}")
            print(execve_line)
            traceback.print_exc()
            command = ''
    else:
        command = ''
            
    # syscall row
    try:
        syscall_row = lines[-1]
        
        syscall_match = re.search(r'syscall=(\d+)', syscall_row)
        if syscall_match:
            syscall = syscall_match.group(1)
        else:
            syscall = 0

        pid_match = re.search(r'pid=(\d+)', syscall_row)
        if pid_match:
            pid = pid_match.group(1)
        else:
            pid = 0

        ppid_match = re.search(r'ppid=(\d+)', syscall_row)
        if ppid_match:
            ppid = ppid_match.group(1)
        else:
            ppid = 0
        
        uid_match = re.search(r'uid=(\d+)', syscall_row)
        if uid_match:
            uid = uid_match.group(1)
        else:
            uid = 0

        euid_match = re.search(r'euid=(\d+)', syscall_row)
        if uid_match:
            euid = uid_match.group(1)
        else:
            euid = 0  

    except Exception as e:
        print(f"Error parsing syscall: {e}")
        traceback.print_exc()
        syscall, pid, ppid, uid, euid = 0, 0, 0, 0, 0
    
    return {"timestamp":timestamp, "cwd":cwd, "arg_count":int(arg_count), "command":command, 
            "syscall":int(syscall), "pid":int(pid), "ppid":int(ppid), "uid":int(uid), "euid":int(euid)}


def aggregate_recent_logs(logs: List[Dict[str, Union[str, int]]]):
    df = pd.DataFrame(logs)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['cwd_depth'] = df['cwd'].str.count('/')  
    
    # sensitive cwd count
    sensitive_paths = ['/etc', '/var', '/root']
    df['is_sensitive_cwd'] = df['cwd'].apply(lambda x: int(any(p in x for p in sensitive_paths)) if isinstance(x, str) else 0)

    sus_keywords = ['rm', 'netcat', 'nmap', 'wget']
    df['has_sus_kw'] = df['command'].apply(lambda cmd: int(any(kw in cmd for kw in sus_keywords)) if isinstance(cmd, str) else 0)

    
    agg = df.groupby('uid').agg(
        unique_command_count=('command', pd.Series.nunique),
        sensitive_cwd_count=('is_sensitive_cwd', 'sum'),
        sus_keyword_count=('has_sus_kw', 'sum'),
        mean_arg_count=('arg_count', 'mean'),
        mean_cwd_depth=('cwd_depth', 'mean'),
        command_count=('command', 'count')
    )
    
    print(agg.reset_index())

def main():
    stdout_output = run_ausearch()
    if not stdout_output:
        print("No output from ausearch.")
        return
    
    logs = [log for log in stdout_output.split('----') if log.strip()]

    parsed_data = [parse_log(log) for log in logs if parse_log(log)]
    
    client = ClientMongo()
    client.write_to_db("bash_logs", parsed_data)
    recent_logs = client.get_data("bash_logs", {"timestamp": {"$gte": int(time.time()) - 5 * 60}})
    aggregate_recent_logs(recent_logs)
    client.close_connection()
    
    with open('logfile.txt', 'a') as f:
        f.write(f"Done writing {len(parsed_data)} logs into db.")
    
main()