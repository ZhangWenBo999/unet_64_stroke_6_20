import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from cleanfid import fid
import lpips
import os

def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output

def FID(input, target):
    with torch.no_grad():
        fid_score = fid.compute_fid(input, target, num_workers=0, verbose=False)
    return fid_score

def LPIPS(input, target, use_gpu=True):
    with torch.no_grad():
        ## Initializing the model
        loss_fn = lpips.LPIPS(net='alex', version=0.1, verbose=False)
        if (use_gpu):
            loss_fn.cuda()

        files = os.listdir(input)

        dist_list = []
        for file in files:
            if (os.path.exists(os.path.join(target, file))):
                # Load images
                img0 = lpips.im2tensor(lpips.load_image(os.path.join(input, file)))  # RGB image from [-1,1]
                img1 = lpips.im2tensor(lpips.load_image(os.path.join(target, file)))

                if (use_gpu):
                    img0 = img0.cuda()
                    img1 = img1.cuda()

                # Compute distance
                dist01 = loss_fn.forward(img0, img1)
                # print('%s: %.3f' % (file, dist01))

                dist_list.append(dist01)

        # print('all-lpips:%.6f\n' % (sum(dist_list) / len(dist_list)))

        lpips_score = (sum(dist_list) / len(dist_list)).detach().cpu().reshape(-1).numpy()
    return lpips_score



def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)