import argparse

import inkid


def main():
    parser = argparse.ArgumentParser()
    inkid.ops.add_subvolume_args(parser)
    args = parser.parse_args()
    print(args)


if __name__ == '__main__':
    main()
