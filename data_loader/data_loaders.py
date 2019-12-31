from base import BaseDataLoader
from data_loader.data_transformations import Normalize, Rescale, RandomCrop, ToTensor
import os
import pandas as pd
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
        key_pts = key_pts.astype('float').reshape(-1, 2)
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
    facial_dataset = FacialKeypointsDataset(csv_file='../data/training_frames_keypoints.csv',
                                            root_dir='../data/training')
    print('Number of images: ', len(facial_dataset))

    # Test dataloader
    data_loader = FacialKeypointsDataLoader(csv_file='../data/training_frames_keypoints.csv',
                                            root_dir='../data/training',
                                            batch_size=4,
                                            shuffle=True,
                                            num_workers=0)
    batch = next(iter(data_loader))
    print("batch['image'].shape     : " + str(batch['image'].shape))
    print("batch['keypoints'].shape : " + str(batch['keypoints'].shape))
