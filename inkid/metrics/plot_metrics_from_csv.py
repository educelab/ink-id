import argparse
import csv
import datetime

from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    args = parser.parse_args()

    dates = []
    metrics = []
    mins = []
    maxs = []

    with open(args.input_csv, newline="") as csvfile:
        csvreader = csv.reader(csvfile)
        rows = [row for row in csvreader]

    for i in range(0, len(rows), 2):
        date = datetime.datetime.strptime(rows[i][0], "%Y-%m-%d")

        values = [float(i) for i in rows[i + 1][5:]]
        metric = values[-1]

        halfway = len(values) // 2
        last_half = values[halfway:]

        dates.append(date)
        metrics.append(metric)
        mins.append(metric - min(last_half))
        maxs.append(max(last_half) - metric)

    fig, ax = plt.subplots()
    ax.errorbar(dates, metrics, yerr=[mins, maxs], fmt="o")
    ax.set_xlabel("Job date")
    ax.set_ylabel("Cross entropy loss")
    plt.show()


if __name__ == "__main__":
    main()
