import glob
import math
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2
from torchvision import transforms as T
from torch.nn import functional as F


def random_crop(image, out_size=(512, 512)):
    h, w = image.shape[:2]
    new_h, new_w = out_size
    top = np.random.randint(0, h - new_h) if h > out_size[0] else 0
    left = np.random.randint(0, w - new_w) if w > out_size[1] else 0
    return image[top:top+new_h, left:left+new_w]    


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, config, train_mode):
        super(ImageNetDataset, self).__init__()
        self.config = config
        self.train_mode = train_mode
        self.data_dir = self.config.data_config.training_data_dir
        
        # get imagenet data
        self.imagenet_df = self.get_imagenet_df()
        train_df = self.imagenet_df.sample(frac=0.8, random_state=200)
        test_df = self.imagenet_df.drop(train_df.index)
        self.df = train_df if self.train_mode==True else test_df

        # arguments for patches for vit
        self.patch_size = self.config.data_config.patch_size
        self.patch_stride = self.config.data_config.patch_stride


    def get_imagenet_df(self):
        # all_img_dir = sorted(os.listdir(self.data_dir))
        all_img_dir = sorted(glob.glob(self.data_dir + '/*'))  # list of imagenet folders
        all_img_names = []
        all_img_labels = []
        for img_dir in all_img_dir:
            img_folder = sorted(os.listdir(img_dir))  # list of images
            for img in img_folder:
                all_img_names.append(os.path.join(img_dir, img))
                all_img_labels.append(all_img_dir.index(img_dir))
        imagenet_df = pd.DataFrame({'img_path':all_img_names, 'label':all_img_labels})
        print(imagenet_df)
        del all_img_names, all_img_labels
        return imagenet_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['img_path'].iloc[idx]
        target = self.df['label'].iloc[idx]
        processed_img, processed_mask = self.prepare_image(img_path)
        return {'image':processed_img, 'mask':processed_mask, 'target':target}

    def prepare_image(self, img_path):
        numpy_image = cv2.imread(img_path)
        cropped_image = random_crop(numpy_image)
        image_tensor = torch.from_numpy(cropped_image)

        # apply random augmentation
        if self.train_mode:
            image = v2.RandAugment(interpolation=T.InterpolationMode.BILINEAR)(image_tensor.permute(2, 0, 1)).permute(1, 2, 0)

        # normalize between -1 to 1:
        image = image/255.0
        image = image*2.0 - 1.0

        # extract patches and masks:
        input_patches = self.extract_patches(image)
        input_patches = input_patches.reshape(input_patches.shape[0], -1, input_patches.shape[-1])
        print(input_patches.shape)
        return input_patches, None

    def extract_patches(self, image):
        """
        similar to tf_extract_patches with 'same' padding.
        refer to discussions on pytorch forum: https://discuss.pytorch.org/t/how-to-extract-smaller-image-patches-3d/16837/9 
        """
        if len(list(image.shape)) == 3:
          image = image.unsqueeze(0).permute(0, 3, 1, 2)
        b, c, h, w = image.shape
        h2 = math.ceil(h / self.patch_stride)
        w2 = math.ceil(w / self.patch_stride)
        dilation = 1
        pad_col = (h2 - 1) * self.patch_stride + (self.patch_size - 1) * dilation + 1 - h
        pad_row = (w2 - 1) * self.patch_stride + (self.patch_size - 1) * dilation + 1 - w
        image = F.pad(image, (pad_row // 2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
        patches = image.unfold(2, self.patch_size, self.patch_stride).unfold(3, self.patch_size, self.patch_stride)
        patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()
        patches = patches.view(b, -1, patches.shape[-2], patches.shape[-1])
        return patches.permute(0, 2, 3, 1)


def get_dataset(config=None):
    if config.pp_config.dataset_name == "imagenet":
        train_dataset = ImageNetDataset(config=config, train_mode=True)
        test_dataset = ImageNetDataset(config=config, train_mode=False)    


if __name__ == "__main__":
    import utils as ut
    import ml_collections
    from torch.utils.data import DataLoader
    
    data_config, model_config = ut.get_params_and_config()
    cfg = ml_collections.ConfigDict({'data_config':data_config, 'model_config':model_config})

    imagedir = "data/imagenet_mimic"
    cfg.data_config.training_data_dir = imagedir
    training_data = ImageNetDataset(cfg, True)
    training_dataloader = DataLoader(training_data, batch_size=2)
    for i, batch in enumerate(training_data):
      print(batch.keys)
