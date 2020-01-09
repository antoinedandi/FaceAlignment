from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


#######################################################################################################################
##### First approach : image to 3D landmarks #####
#######################################################################################################################

class First3DFaceAlignmentModel(BaseModel):

    def __init__(self):
        super(First3DFaceAlignmentModel, self).__init__()

        # input image : 1 x 224 x 224, grayscale squared images

        self.conv1 = nn.Conv2d(1, 32, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(128, 256, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(43264, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.dropout5 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn6 = nn.BatchNorm1d(1000)
        self.dropout6 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(1000, 3*68)

        # I.xavier_uniform(self.fc1.weight.data)
        # I.xavier_uniform(self.fc2.weight.data)
        # I.xavier_uniform(self.fc3.weight.data)

    def forward(self, x):

        # Convolutions
        x = self.dropout1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(F.elu(self.bn4(self.conv4(x)))))

        # flatten
        x = x.view(x.size(0), -1)

        x = self.dropout5(F.elu(self.bn5(self.fc1(x))))
        x = self.dropout6(F.elu(self.bn6(self.fc2(x))))
        x = self.fc3(x)

        return x


#######################################################################################################################
##### Second approach : image to 2D landmarks to 3D landmarks #####
#######################################################################################################################


class FaceAlignmentModel2D(BaseModel):

    def __init__(self):
        super(FaceAlignmentModel2D, self).__init__()

        # input image : 1 x 224 x 224, grayscale squared images

        self.conv1 = nn.Conv2d(1, 32, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(128, 256, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(43264, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.dropout5 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn6 = nn.BatchNorm1d(1000)
        self.dropout6 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(1000, 2*68)

        # I.xavier_uniform(self.fc1.weight.data)
        # I.xavier_uniform(self.fc2.weight.data)
        # I.xavier_uniform(self.fc3.weight.data)

    def forward(self, x):

        # Convolutions
        x = self.dropout1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(F.elu(self.bn4(self.conv4(x)))))

        # flatten
        x = x.view(x.size(0), -1)

        x = self.dropout5(F.elu(self.bn5(self.fc1(x))))
        x = self.dropout6(F.elu(self.bn6(self.fc2(x))))
        x = self.fc3(x)

        return x


class Model2Dto3DLandmarks(BaseModel):

    def __init__(self):
        super(Model2Dto3DLandmarks, self).__init__()

        # input (bs , 68 * 2) - 2D landmarks

        self.fc1 = nn.Linear(2*68, 2*68)
        self.bn1 = nn.BatchNorm1d(2*68)
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(2 * 68, 2 * 68)
        self.bn2 = nn.BatchNorm1d(2 * 68)
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(2 * 68, 2 * 68)
        self.bn3 = nn.BatchNorm1d(2 * 68)
        self.dropout3 = nn.Dropout(p=0.3)

        self.fc4 = nn.Linear(2 * 68, 2 * 68)
        self.bn4 = nn.BatchNorm1d(2 * 68)
        self.dropout4 = nn.Dropout(p=0.3)

        self.fc5 = nn.Linear(2 * 68, 2 * 68)
        self.bn5 = nn.BatchNorm1d(2 * 68)
        self.dropout5 = nn.Dropout(p=0.3)

        self.fc6 = nn.Linear(2*68, 68)

    def forward(self, x):

        x = self.dropout1(F.elu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.elu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.elu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.elu(self.bn4(self.fc4(x))))
        x = self.dropout5(F.elu(self.bn5(self.fc5(x))))
        x = self.fc6(x)

        return x


class Second3DFaceAlignmentModel(BaseModel):

    def __init__(self):
        super(Second3DFaceAlignmentModel, self).__init__()

        self.face_alignment_2D = FaceAlignmentModel2D()
        self.model_2D_to_3D = Model2Dto3DLandmarks()

    def forward(self, x):

        x1 = self.face_alignment_2D(x)
        x2 = self.model_2D_to_3D(x1)
        x3 = torch.cat([x1, x2], dim=1)

        return x3
