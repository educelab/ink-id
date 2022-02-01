import argparse

import inkid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('volume')
    inkid.util.add_subvolume_args(parser)
    args = parser.parse_args()



if __name__ == '__main__':
    main()
