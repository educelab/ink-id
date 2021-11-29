import argparse

from inkid.data import PPM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('x', type=int)
    parser.add_argument('y', type=int)
    parser.add_argument('z', type=int)
    parser.add_argument('output')
    args = parser.parse_args()

    ppm = PPM.from_path(args.input)
    ppm.translate(args.x, args.y, args.z)
    ppm.write(args.output)


if __name__ == '__main__':
    main()
