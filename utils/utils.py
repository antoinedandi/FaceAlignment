import json
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import cv2
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


#######################################################################################################################
##### Project utils #####
#######################################################################################################################

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


#######################################################################################################################
##### Visualization utils #####
#######################################################################################################################

"""
def show_all_keypoints_2D(image, predicted_key_pts, gt_pts=None, axis=None):
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')
        if axis is not None:
            for i in range(len(gt_pts[:, 0])):
                axis.annotate(str(i), (gt_pts[i, 0], gt_pts[i, 1]))

def show_all_keypoints_3D(image, predicted_key_pts, gt_pts=None, axis=None):
    return 0


def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):

    # Defining the figure
    plt.figure(figsize=(5 * batch_size, 5))

    for i in range(batch_size):
        ax = plt.subplot(1, batch_size, i + 1)
        image = test_images[i].data
        image = image.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.cpu().numpy()
        predicted_key_pts = predicted_key_pts * 50.0 + 100

        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts * 50.0 + 100

        show_all_keypoints_2D(np.squeeze(image), predicted_key_pts, ground_truth_pts)

        plt.axis('off')

    plt.show()
"""


def visualize_keypoints(image, predicted_kp, true_kp=None):

    image = np.squeeze(image.numpy())
    predicted_kp = predicted_kp.cpu().numpy()
    predicted_kp = predicted_kp * 50.0 + 100

    # Visualize Image
    fig = plt.figure(figsize=(20, 4))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(image, cmap='gray')
    ax.scatter(predicted_kp[:, 0], predicted_kp[:, 1], s=20, marker='.', c='m')
    ax.axis('off')

    if true_kp is not None:
        true_kp = true_kp * 50.0 + 100
        ax.scatter(true_kp[:, 0], true_kp[:, 1], s=20, marker='.', c='g')

    # Visualize 3D keypoints
    x = - predicted_kp[:, 0]
    y = predicted_kp[:, 1]
    z = predicted_kp[:, 2]

    # First 3D view
    ax = fig.add_subplot(1, 4, 2, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.view_init(elev=95., azim=90.)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Second 3D view
    ax = fig.add_subplot(1, 4, 3, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.view_init(elev=120., azim=60.)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Third 3D view
    ax = fig.add_subplot(1, 4, 4, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.view_init(elev=60., azim=60.)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.tight_layout()
    plt.show()


def visualize_output(images, predicted_kps, true_kps):
    batch_size = images.shape[0]
    for i in range(batch_size):
        ground_truth_pts = None
        if true_kps is not None:
            ground_truth_pts = true_kps[i]
        visualize_keypoints(images[i], predicted_kps[i], ground_truth_pts)


def roi_transform(roi, output_size=224):
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    roi = roi / 255.0
    roi = cv2.resize(roi, (output_size, output_size))
    roi_tensor = torch.from_numpy(roi.reshape(1,1,output_size,output_size))
    roi_tensor = roi_tensor.type(torch.FloatTensor)
    return roi, roi_tensor

