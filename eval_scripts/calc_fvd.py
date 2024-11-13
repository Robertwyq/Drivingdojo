
import os, random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from pytorch_i3d import InceptionI3d

from sklearn.metrics.pairwise import polynomial_kernel


MAX_BATCH = 16
TARGET_RESOLUTION = (224, 224)

def preprocess(videos, target_resolution):
    # videos in {0, ..., 255} as np.uint8 array
    b, t, h, w, c = videos.shape
    all_frames = torch.FloatTensor(videos).flatten(end_dim=1) # (b * t, h, w, c)
    all_frames = all_frames.permute(0, 3, 1, 2).contiguous() # (b * t, c, h, w)
    resized_videos = F.interpolate(all_frames, size=target_resolution,
                                   mode='bilinear', align_corners=False)
    resized_videos = resized_videos.view(b, t, c, *target_resolution)
    output_videos = resized_videos.transpose(1, 2).contiguous() # (b, c, t, *)
    scaled_videos = 2. * output_videos / 255. - 1 # [-1, 1]
    return scaled_videos

def get_fvd_logits(videos, i3d, device):
    videos = preprocess(videos, TARGET_RESOLUTION)
    embeddings = get_logits(i3d, videos, device)
    return embeddings

def load_fvd_model(device):
    i3d = InceptionI3d(400, in_channels=3).to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    i3d_path = os.path.join(current_dir, 'i3d_pretrained_400.pt')
    print("=== Loading I3D from {} ===".format(i3d_path))
    i3d.load_state_dict(torch.load(i3d_path, map_location=device))
    i3d.eval()
    return i3d


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())

# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1) # unbiased estimate
    m_center = m - torch.mean(m, dim=1, keepdim=True)
    mt = m_center.t()  # if complex: mt = m.t().conj()
    return fact * m_center.matmul(mt).squeeze()


def polynomial_mmd(X, Y):
    m = X.shape[0]
    n = Y.shape[0]
    # compute kernels
    K_XX = polynomial_kernel(X)
    K_YY = polynomial_kernel(Y)
    K_XY = polynomial_kernel(X, Y)
    # compute mmd distance
    K_XX_sum = (K_XX.sum() - np.diagonal(K_XX).sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - np.diagonal(K_YY).sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)
    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum
    return mmd


def get_logits(i3d, video_loader, device):
    with torch.no_grad():
        logits = []
        for i, batch in enumerate(tqdm(video_loader)):
            logits.append(i3d(batch.to(device).transpose(1,2)))
        logits = torch.cat(logits, dim=0)
        return logits

class VideoDataset(Dataset):
    def __init__(self, video_folder, n_frames=16, resolution=(224,224)):
        self.videos = []
        for x in os.listdir(video_folder):
            d = os.path.join(video_folder, x)

            if os.path.isdir(d) and len(os.listdir(d)) >= n_frames: self.videos.append(d)

        self.n_frames = n_frames
        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((resolution), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            ]
        )

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        transformed_frames = []

        frames = sorted(os.listdir(self.videos[index]))

        start_id = random.randint(0, len(frames) - self.n_frames)
        for img in frames[start_id:start_id + self.n_frames]:
            transformed_frames.append(self.img_transform(Image.open(os.path.join(self.videos[index], img)).convert('RGB')))
        
        video_tensor = torch.stack(transformed_frames) * 2 - 1.0

        return video_tensor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_path', type=str, default='fake data path')
    parser.add_argument('--real_path', type=str, default='real data dir')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--n_runs', type=int, default=5, help='calculate multiple times')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_sample', type=int, default=1024)
    parser.add_argument('--use_cache', action="store_true", default=False, help="Whether or not to use the preprocessed features of the real data")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(args)
    
    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")

    cache_dir = os.path.join(os.path.dirname(__file__), "cached")

    os.makedirs(cache_dir, exist_ok=True)

    # Load real videos
    real_data = VideoDataset(args.real_path, args.num_frames, TARGET_RESOLUTION)
    real_data_loader = DataLoader(real_data, batch_size=args.batch_size, num_workers=8)

    # Load fake videos
    fake_data = VideoDataset(args.fake_path, args.num_frames, TARGET_RESOLUTION)
    fake_data_loader = DataLoader(fake_data, batch_size=args.batch_size, num_workers=8)

    # Compute FVD
    i3d = load_fvd_model(device)

    FVDs = []
    for i in range(args.n_runs):
        print(f"\n=============== Run {i} (Use cache feat. of real data: {args.use_cache}) =====================")
        print(f"Processing real data ({len(real_data.videos)}) videos...")

        cache_path = os.path.join(cache_dir, os.path.basename(args.real_path) + ".npz")
        if args.use_cache:
            print("Loading preprocessed real data statistics from ", cache_path)
            cached = np.load(cache_path)
            m, sigma = torch.tensor(cached['m'], device=device), torch.tensor(cached['sigma'], device=device)
        else:
            real_embeddings = get_logits(i3d, real_data_loader, device)
            m = real_embeddings.flatten(start_dim=1).mean(dim=0)
            sigma = cov(real_embeddings, rowvar=False)

        if args.use_cache and not os.path.exists(cache_path):
            print("Saving processed real data statistics to ", cache_path)
            np.savez(cache_path, m=m.cpu().numpy(), sigma=sigma.cpu().numpy())

        print(f"Processing fake data ({len(fake_data.videos)}) videos...")
        fake_embeddings = get_logits(i3d, fake_data_loader, device)
        m_w = fake_embeddings.flatten(start_dim=1).mean(dim=0)
        sigma_w = cov(fake_embeddings, rowvar=False)

        sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
        trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

        mean = torch.sum((m - m_w) ** 2)

        FVD = trace + mean
    
        print("FVD: ", FVD.item())

        FVDs.append(FVD.item())
    
    mean, std = np.mean(FVDs), np.std(FVDs)
    print("Real path:{}\nFake path:{}".format(args.real_path, args.fake_path))
    print("{} runs FVD: {:.2f} ({:.2f})".format(args.n_runs, mean, std))