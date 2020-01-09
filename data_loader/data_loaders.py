from base import BaseDataLoader
from data_loader.data_transformations import Normalize, Rescale, RandomCrop, ToTensor
from utils import visualize_keypoints
import os
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from torch.utils.data import Dataset
from torchvision import transforms


class FacialKeypointsDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

        # Data transform
        transform = transforms.Compose([Rescale(250),
                                        RandomCrop(224),
                                        Normalize(),
                                        ToTensor()])

        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                  self.key_pts_frame.iloc[idx, 0])
        image = mpimg.imread(image_name)

        if (image.shape[2] == 4):
            image = image[:, :, 0:3]

        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 3)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample


class FacialKeypointsDataLoader(BaseDataLoader):
    def __init__(self, csv_file, root_dir, batch_size=4, shuffle=True, validation_split=0.0,
                 num_workers=0):
        self.dataset = FacialKeypointsDataset(csv_file=csv_file,
                                              root_dir=root_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == '__main__':
    data_directory = '../data'

    # Test dataset
    facial_dataset = FacialKeypointsDataset(csv_file='../data/training_keypoints.csv',
                                            root_dir='../data/training')
    print('Number of images: ', len(facial_dataset))
    print(facial_dataset[1])

    # Draw 3D face
    sample = facial_dataset[1]
    image = sample['image']
    key_pts = sample['keypoints']
    visualize_keypoints(image, key_pts)

    """
    image = np.squeeze(image.numpy())
    key_pts = key_pts.cpu().numpy()
    key_pts = key_pts * 50.0 + 100

    predicted_key_points = key_pts + [5., 5., 0]

    # Image
    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(image, cmap='gray')
    ax.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='g')
    ax.scatter(predicted_key_points[:, 0], predicted_key_points[:, 1], s=20, marker='.', c='m')
    ax.axis('off')

    # 3D
    x = - key_pts[:, 0]
    y = key_pts[:, 1]
    z = key_pts[:, 2]
    ax = fig.add_subplot(2, 1, 2, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')

    ax.view_init(elev=95., azim=90.)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.show()
    """

    # Test dataloader
    data_loader = FacialKeypointsDataLoader(csv_file='../data/training_keypoints.csv',
                                            root_dir='../data/training',
                                            batch_size=4,
                                            shuffle=True,
                                            num_workers=0)
    batch = next(iter(data_loader))
    print("batch['image'].shape     : " + str(batch['image'].shape))
    print("batch['keypoints'].shape : " + str(batch['keypoints'].shape))

