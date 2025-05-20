CONFIG_PATH = '/etc/vsd-linux.conf'


def set_ip(ip):
    with open(CONFIG_PATH, 'w') as f:
        f.write(f"IP={ip}\n")
    print(f"Set IP to {ip}")
