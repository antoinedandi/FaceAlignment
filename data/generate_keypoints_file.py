import glob
import csv
from numpy import loadtxt

keypoints_fn = glob.glob("keypoints_test/*.txt")

with open('test_keypoints.csv', mode='w') as keypoints_file:
    keypoints_writer = csv.writer(keypoints_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    first_row = [''] + [i for i in range(3*68)]
    keypoints_writer.writerow(first_row)

    for fn in keypoints_fn:
        vec = loadtxt(fn)
        vec = vec.transpose().reshape(-1)
        row = [fn.replace('keypoints_training/', '').replace('keypoints_test/', '').replace('.txt', '.jpg')] \
              + [vec[i] for i in range(len(vec))]
        keypoints_writer.writerow(row)

