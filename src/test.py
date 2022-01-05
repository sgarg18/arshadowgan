# -*- coding: utf-8 -*-

import cv2
import os
import os.path as osp
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import ARDataset,ARDataset_without_ref
from networks import ARShadowGAN,Generator_with_Refin
from utils import get_validation_augmentation, get_preprocessing


def get_parser():
    """A parser for command line arguments """

    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for test')
    parser.add_argument('--img_size_h', type=int, default=256,
                        help='image size')
    parser.add_argument('--img_size_w', type=int, default=256,
                        help='image size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers for test')
    parser.add_argument('--dataset_path', type=str,
                        help='path to dataset')
    parser.add_argument('--result_path', type=str,
                        help='path to results')
    parser.add_argument('--attention_encoder', type=str, default='resnet34',
                        help='encoder for attention block')
    parser.add_argument('--SG_encoder', type=str, default='resnet18',
                        help='encoder for shadow generation block')
    parser.add_argument('--path_att', type=str,
                        help='path to attention block weights')
    parser.add_argument('--path_SG', type=str,
                        help='path to shadow generation block weights')
    parser.add_argument('--path_gen', type=str,
                        help='path to shadow generation block weights')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='encoder for attention part (default: resnet34)')
    return parser

def load_ckp(checkpoint_fpath,model):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    # checkpoint = torch.load(checkpoint_fpath)
    checkpoint = torch.load(checkpoint_fpath,map_location='cpu')

    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    
    return model

def test_arshadowgan():
    """Test ARShadowGAN-like architecture """
    # parse command line arguments
    args = get_parser().parse_args()

    # create folders
    if not osp.exists(args.result_path):
        os.makedirs(args.result_path)

    # dataset and dataloader declaration
    dataset = ARDataset(args.dataset_path, augmentation=get_validation_augmentation(args.img_size), preprocessing=get_preprocessing(), is_train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # define device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    device='cpu'
    # define model
    model = ARShadowGAN(
        encoder_att=args.attention_encoder,
        encoder_SG=args.SG_encoder,
        model_path_attention=args.path_att,
        model_path_SG=args.path_SG,
        device=device
    )

    # model in eval mode now
    model.eval()

    # inference
    counter = 0
    for i, data in enumerate(tqdm(dataloader)):
        tensor_att = torch.cat((data[0][:, :3], torch.unsqueeze(data[1][:, -1], axis=1)), axis=1).to(device)
        tensor_SG = torch.cat((data[2][:, :3], torch.unsqueeze(data[3][:, -1], axis=1)), axis=1).to(device)

        with torch.no_grad():
            result, output_mask1 = model(tensor_att, tensor_SG)

        for j in range(args.batch_size):
            counter += 1
            output_image = np.uint8(127.5 * (result.cpu().numpy()[j].transpose(1,2,0) + 1.0))
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(osp.join(args.result_path, str(counter) + '.png'), output_image)



def shadowgan():
    """Test ARShadowGAN-like architecture """
    # parse command line arguments
    args = get_parser().parse_args()

    # create folders
    if not osp.exists(args.result_path):
        os.makedirs(args.result_path)

    # dataset and dataloader declaration
    img_size=[args.img_size_h,args.img_size_w]
    dataset = ARDataset_without_ref(args.dataset_path, augmentation=get_validation_augmentation(img_size), preprocessing=get_preprocessing(), is_train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # define device
    device =torch.device(args.device if torch.cuda.is_available() else 'cpu')
    device='cpu'

    # define model
    model =  Generator_with_Refin(args.encoder)
    model.to(device)

    model = load_ckp(args.path_gen,model)


    # model in eval mode now
    model.eval()

    # inference
    counter = 0
    for i, data in enumerate(tqdm(dataloader)):
        noshadow_image = data[2][:, :3].to(device)
        mask = torch.unsqueeze(data[3][:, 0], 1).to(device)
        # bg_mask = torch.unsqueeze(data[3][:, 1], 1).to(device)
        model_input = torch.cat((noshadow_image, mask), axis=1)

        with torch.no_grad():
            shadow_mask_tensor1,shadow_mask_tensor2 = model(model_input)
            result = torch.add(noshadow_image, shadow_mask_tensor2)

        for j in range(args.batch_size):
            counter += 1
            image=result.cpu().numpy()[j].transpose(1,2,0)
            image=(image-np.min(image))/(np.max(image)-np.min(image))
            output_image = np.uint8(255 * (image))

            # output_image = np.uint8(127.5 * (result.cpu().numpy()[j].transpose(1,2,0) + 1.0))

            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(osp.join(args.result_path, str(counter) + '.png'), output_image)
        if counter>=30:
            break

if __name__ == "__main__":
    shadowgan()
