import argparse

import inkid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='directory to upload')
    parser.add_argument('--rclone-transfer-remote', metavar='remote', default=None,
                        help='if specified, and if matches the name of one of the directories in '
                        'the output path, transfer the results to that rclone remote into the '
                        'sub-path following the remote name')
    args = parser.parse_args()

    inkid.ops.rclone_transfer_to_remote(args.rclone_transfer_remote, args.dir)


if __name__ == '__main__':
    main()
