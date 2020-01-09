import argparse
import torch
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from utils.utils import visualize_output


def main(config):

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        csv_file="data/training_keypoints.csv",
        root_dir="data/training",
        batch_size=8,
        shuffle=True,
        validation_split=0.1,
        num_workers=0)

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

        sample = next(iter(data_loader))

        # Inference
        images = sample['image']
        images = images.type(torch.FloatTensor).to(device)    # images :  (bs * 1 * 224 * 224)
        key_pts = sample['keypoints']
        key_pts = key_pts.view(key_pts.size(0), 68, -1)
        key_pts = key_pts.type(torch.FloatTensor).to(device)  # key_pts : (bs * 68 * 2)
        outputs = model(images)
        outputs = outputs.view(outputs.size(0), 68, -1)       # outputs : (bs * 68 * 2)

        # Output Visualization
        visualize_output(images, outputs, key_pts)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
