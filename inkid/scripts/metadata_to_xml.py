import argparse
import json

import inkid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()

    with open(args.infile) as f:
        data = json.load(f)
    xml = inkid.ops.dict_to_xml(data)
    with open(args.outfile, 'w') as f:
        f.write(xml)


if __name__ == '__main__':
    main()