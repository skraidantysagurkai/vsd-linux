import argparse

from vsd_linux.monitor import start_monitor, stop_monitor


def main():
    parser = argparse.ArgumentParser(prog='vsd-linux')
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser('help')
    subparsers.add_parser('start')
    subparsers.add_parser('stop')
    subparsers.add_parser('status')

    config_parser = subparsers.add_parser('config')
    config_parser.add_argument('-ip', required=True)

    args = parser.parse_args()

    if args.command == 'help':
        parser.print_help()
    elif args.command == 'start':
        start_monitor()
    elif args.command == 'stop':
        stop_monitor()
    # elif args.command == 'status':
    #     status_monitor()
    # elif args.command == 'config':
    #     set_ip(args.ip)
    print("Command executed successfully.")


if __name__ == '__main__':
    main()
