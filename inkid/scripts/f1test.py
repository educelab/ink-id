import argparse

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=str)
    parser.add_argument("mask", type=str)
    parser.add_argument("prediction", type=str)
    args = parser.parse_args()

    ground_truth = Image.open(args.ground_truth).convert("L")
    mask = Image.open(args.mask).convert("L")
    prediction = Image.open(args.prediction).convert("L")

    assert ground_truth.size == prediction.size == mask.size, "Image sizes do not match"

    for threshold in range(111, 255, 10):
        prediction = Image.open(args.prediction).convert("L")
        # p_array = np.array(prediction)
        # print(p_array.shape)
        # print(p_array.dtype)
        # print(p_array.min())
        # print(p_array.max())
        # print(p_array.mean())
        thresholded_prediction = prediction.point(lambda v: 0 if v < threshold else 255)
        # thresholded_prediction = thresholded_prediction.convert("1")
        # thresholded_prediction.show()

        assert ground_truth.mode == thresholded_prediction.mode == mask.mode, "Image modes do not match"

        # calculate the F1 score
        tp = 0
        fp = 0
        fn = 0

        for y in range(ground_truth.size[1]):
            for x in range(ground_truth.size[0]):
                if mask.getpixel((x, y)) != 0:
                    if thresholded_prediction.getpixel((x, y)) != 0:
                        if ground_truth.getpixel((x, y)) != 0:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        if ground_truth.getpixel((x, y)) != 0:
                            fn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        print(f"Threshold: {threshold} F1 score: {f1}")


if __name__ == "__main__":
    main()
