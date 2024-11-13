"""Calculates the Frechet Inception Distance (FID)
The calculation pipeline follows the appendix of "Align Your Latents: High-Resolution Video Synthesis With Latent Diffusion Models"

Code apapted from https://github.com/mseitzer/pytorch-fid
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import random
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from inception import InceptionV3


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, folder, num_images = 1e4):
        video_dirs = [os.path.join(folder, x) for x in os.listdir(folder)]

        image_paths = []
        for video_dir in video_dirs:
            if os.path.isdir(video_dir):
                image_paths.extend([os.path.join(video_dir, x) for x in os.listdir(video_dir)])
        
        self.image_paths = random.sample(image_paths, int(num_images))

        # Resizing and scaling are done in the Inception model. No need here.
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        img = Image.open(self.image_paths[i]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(folder, model, batch_size=50, dims=2048, device='cpu',
                    num_images=1e4, num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- folder       : A folder containing videos, each of which is a folder of images (frames)
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    dataset = VideoDataset(folder, num_images)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = []

    print("Processing ", folder)
    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr.append(pred)
    return np.concatenate(pred_arr, axis=0)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(folder, model, batch_size=50, dims=2048,
                                    device='cpu', num_images=1e4, num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- folder       : the real video folder or the fake one
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(folder, model, batch_size, dims, device, num_images, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma



def calculate_fid_given_paths(paths, batch_size, device, dims, num_images=1e4, num_workers=1, ckpt_path=''):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], ckpt_path=ckpt_path).to(device)

    m1, s1 = calculate_activation_statistics(paths[0], model, batch_size,
                                        dims, device, num_images, num_workers)
    m2, s2 = calculate_activation_statistics(paths[1], model, batch_size,
                                        dims, device, num_images, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int,
                        help=('Number of processes to use for data loading. '
                            'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                            'By default, uses pool3 features'))
    parser.add_argument('--inception_ckpt', type=str,
                        default=os.path.join(os.path.dirname(__file__), "pretrained/pt_inception-2015-12-05-6726825d.pth"),
                        help="The path to the inception-v3 pretrained model")

    parser.add_argument('path', type=str, nargs=2,
                        help=('Paths to the generated images or '
                            'to .npz statistic files'))
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--n_frames', type=int, default=10000)

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    device = torch.device(f'cuda:{args.gpu}' )

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    FIDs = []
    for i in range(args.n_runs):
        print(f"\n=============== Run {i + 1}: =====================")

        fid_value = calculate_fid_given_paths(args.path,
                                            args.batch_size,
                                            device,
                                            args.dims,
                                            args.n_frames,
                                            num_workers,
                                            args.inception_ckpt)
        print('Run ', i, 'FID: ', fid_value)

        FIDs.append(fid_value)
    
    print("Calculate FID between {} and {}\n FID: {:.2f} ({:.2f})".format(args.path[0], args.path[1], np.mean(FIDs), np.std(FIDs)))


if __name__ == '__main__':
    main()