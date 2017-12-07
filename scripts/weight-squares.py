import tifffile as tiff
import numpy as np

mask = tiff.imread('small-fragment-outline.tif')
ink = tiff.imread('small-fragment-gt.tif')
total_ink = np.count_nonzero(ink)
rowspan = int(mask.shape[0] / 5)
colspan = int(mask.shape[1] / 2)
print("Rowspan: {}, Colspan: {}".format(rowspan, colspan))

precs = [ 54.2,
        37.5,
        77.5,
        93.7,
        92.3,
        92.6,
        22.2,
        93.3,
        82.2,
        91.7
        ]
recs = [22.9,
        87.3,
        80.8,
        91.3,
        74.5,
        77.9,
        62.8,
        60.8,
        69.4,
        81.1
        ]

prec_weights = 0.0
rec_weights = 0.0
weighted_prec = 0.0
weighted_rec = 0.0

for i in range(10):
    print("Working on square {}".format(i))
    col = (i % 2) * colspan
    row = int(i/2) * rowspan
    square = mask[row:row+rowspan, col:col+colspan]
    ink_in_square = np.count_nonzero(ink[row:row+rowspan, col:col+colspan])
    fraction = float(ink_in_square) / float(total_ink)
    print(" number on fragment: {}".format(np.count_nonzero(square)))
    print(" number of ink: {}".format(ink_in_square))
    print(" fraction of ink: {:.4f}".format(fraction))
    weighted_prec += fraction*precs[i]
    weighted_rec += fraction*recs[i]
    prec_weights += fraction
    rec_weights += fraction
    
print("\n\nTotal weighted precision: {}".format(weighted_prec))
print("Total weighted recall: {}".format(weighted_rec))
print("(Weight sums: {} and {}".format(prec_weights, rec_weights))
