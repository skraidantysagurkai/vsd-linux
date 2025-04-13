import subprocess
import json
import re

OUTPUT_FILE = "audit_logs.json"


def parse_audit_log(log_lines):
	"""Parses ausearch log lines into structured data."""
	parsed_logs = []
	current_log = {}

	for line in log_lines:
		line = line.strip()

		# Extract timestamp
		if line.startswith("time->"):
			current_log["timestamp"] = line.split("time->")[1]

		# Extract executed command details (proctitle)
		elif "type=PROCTITLE" in line:
			match = re.search(r'proctitle=(.+)', line)
			if match:
				proctitle = bytes.fromhex(match.group(1)).decode(errors='ignore')
				current_log["command"] = proctitle

		# Extract file paths involved
		elif "type=PATH" in line:
			match = re.search(r'name="([^"]+)"', line)
			if match:
				if "paths" not in current_log:
					current_log["paths"] = []
				current_log["paths"].append(match.group(1))

		# Extract current working directory
		elif "type=CWD" in line:
			match = re.search(r'cwd="([^"]+)"', line)
			if match:
				current_log["cwd"] = match.group(1)

		# Extract execution details
		elif "type=EXECVE" in line:
			match = re.findall(r'a\d+="([^"]+)"', line)
			if match:
				current_log["exec_args"] = match

		# Extract syscall details (PID, command name, executable path)
		elif "type=SYSCALL" in line:
			match = re.search(r'pid=(\d+).*comm="([^"]+)".*exe="([^"]+)"', line)
			if match:
				current_log["pid"] = match.group(1)
				current_log["comm"] = match.group(2)
				current_log["exe"] = match.group(3)

		# When encountering a new log entry, save the previous one
		elif "type=" in line and current_log:
			parsed_logs.append(current_log)
			current_log = {}

	if current_log:
		parsed_logs.append(current_log)  # Add last entry if exists

	return parsed_logs


def watch_audit_logs():
	"""Continuously watches the ausearch logs and saves structured data to JSON."""
	process = subprocess.Popen(
		["sudo", "ausearch", "-k", "commands", "-i", "-m", "EXECVE"],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True
	)

	logs = []
	try:
		for line in process.stdout:
			if "----" in line:  # Separator for new log entry
				if logs:
					parsed_logs = parse_audit_log(logs)
					if parsed_logs:
						with open(OUTPUT_FILE, "a") as f:
							json.dump(parsed_logs, f, indent=4)
							f.write("\n")
					logs = []  # Reset logs for new entry
			else:
				logs.append(line)

	except KeyboardInterrupt:
		print("\nStopping log monitoring...")
		process.terminate()


if __name__ == "__main__":
	print("Watching audit logs... Press Ctrl+C to stop.")
	watch_audit_logs()