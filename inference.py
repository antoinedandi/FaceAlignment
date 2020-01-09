import argparse
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import model.model as module_arch
from parse_config import ConfigParser
from utils import visualize_output, roi_transform


def main(config, file):

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model architecture
    model = config.init_obj('arch', module_arch)

    print('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    with torch.no_grad():

        # Show image
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()

        # Detect all faces in an image
        face_cascade = cv2.CascadeClassifier('saved/cv2_models/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=2)
        image_with_detections = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (255, 0, 0), 3)
        plt.imshow(image_with_detections)
        plt.show()

        # Keypoint detection
        image_copy = np.copy(image)
        pad_scale = 0.33
        for i, (x, y, w, h) in enumerate(faces):
            ax = plt.subplot(2, 6, i + 1)
            wpad = int(w * pad_scale)
            hpad = int(h * pad_scale)
            roi = image_copy[y - hpad:y + h + hpad, x - wpad:x + w + wpad]
            roi_norm, roi_tensor = roi_transform(roi)
            roi_tensor = roi_tensor.to(device)
            output_pts = model(roi_tensor)
            torch.squeeze(output_pts)
            output_pts = output_pts.view(68, -1)
            predicted_key_pts = output_pts.data.cpu().numpy()
            predicted_key_pts = predicted_key_pts * 50.0 + 100
            plt.imshow(roi_norm, cmap='gray')
            plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=60, marker='.', c='m')
        plt.show()

        """
        # Inference
        images = data['image']
        images = images.type(torch.FloatTensor).to(device)    # images :  (bs * 1 * 224 * 224)
        key_pts = data['keypoints']
        key_pts = key_pts.view(key_pts.size(0), 68, -1)
        key_pts = key_pts.type(torch.FloatTensor).to(device)  # key_pts : (bs * 68 * 2)
        outputs = model(images)
        outputs = outputs.view(outputs.size(0), 68, -1)       # outputs : (bs * 68 * 2)

        # Output Visualization
        batch_size = images.shape[0]
        visualize_output(images, outputs, key_pts, batch_size)
        """

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-f', '--file', default=None, type=str,
                      help='path to the picture (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')



    config = ConfigParser.from_args(args)
    args = args.parse_args()
    file = args.file
    main(config, file)
