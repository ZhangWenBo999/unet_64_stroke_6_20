import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

from mask_generation.utils import MaskGeneration, MergeMask, RandomAttribute

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        # images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-16')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    # def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[64, 64], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        # ***************** 区分train与evaluate **************
        self.phase = self.mask_config['phase']
        self.m12_mode = self.mask_config["m12_mode"]
        self.img = self.imgs[0]

    def __getitem__(self, index):
        # ***************** 切换train与evaluate **************
        self.phase = self.mask_config['phase']

        # ***************** 待验证 ***********************
        if self.phase == 'eval':
            self.mask_mode = 'center'
        else:
            self.mask_mode = self.mask_config['mask_mode']

        if self.mask_mode == 'm3':
            self.mask_mode = np.random.choice(['ThickStrokes', 'MediumStrokes', 'ThinStrokes'])
        elif self.mask_mode == 'm5':
            self.mask_mode = np.random.choice(['ThickStrokes', 'MediumStrokes', 'ThinStrokes', 'Every_N_Lines', 'Completion'])
        elif self.mask_mode == 'm7':
            self.mask_mode = np.random.choice(['ThickStrokes', 'MediumStrokes', 'ThinStrokes', 'Every_N_Lines', 'Completion', 'Expand', 'Nearest_Neighbor'])

        if self.mask_mode == 'ThinStrokes':
            self.m12_mode = RandomAttribute('ThinStrokes', 256)
        elif self.mask_mode == 'MediumStrokes':
            self.m12_mode = RandomAttribute('MediumStrokes', 256)
        elif self.mask_mode == 'ThickStrokes':
            self.m12_mode = RandomAttribute('ThickStrokes', 256)
        elif self.mask_mode == 'Every_N_Lines':
            self.m12_mode = RandomAttribute('Every_N_Lines', 256)
        elif self.mask_mode == 'Completion':
            self.m12_mode = RandomAttribute('Completion', 256)
        elif self.mask_mode == 'Expand':
            self.m12_mode = RandomAttribute('Expand', 256)
        elif self.mask_mode == 'Nearest_Neighbor':
            self.m12_mode = RandomAttribute('Nearest_Neighbor', 256)

        print(self.mask_mode)


        # if self.phase == 'train':
        #     self.mask_mode = self.mask_config['mask_mode']
        # elif self.phase == 'test':
        #     self.mask_mode = self.mask_config['mask_mode']
        # else:
        #     self.mask_mode = 'center'

        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        mask_generation = MaskGeneration()
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask

            from torchvision.utils import save_image
            mask1 = mask
            mask1 = torch.from_numpy(mask1).squeeze(2)
            save_image(mask1 * 1., "hybrid_mask.png")

        elif self.mask_mode == 'ThinStrokes':
            mask = mask_generation(self.img, self.m12_mode)
        elif self.mask_mode == 'MediumStrokes':
            mask = mask_generation(self.img, self.m12_mode)
        elif self.mask_mode == 'ThickStrokes':
            mask = mask_generation(self.img, self.m12_mode)
        elif self.mask_mode == 'Every_N_Lines':
            mask = mask_generation(self.img, self.m12_mode)
        elif self.mask_mode == 'Completion':
            mask = mask_generation(self.img, self.m12_mode)
        elif self.mask_mode == 'Expand':
            mask = mask_generation(self.img, self.m12_mode)
        elif self.mask_mode == 'Nearest_Neighbor':
            mask = mask_generation(self.img, self.m12_mode)
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


