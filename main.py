import torch, argparse, time
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from metrics import *
from models import *
from datasets import *

parser = argparse.ArgumentParser(description='Inpainting Error Maximization')
parser.add_argument('data_path', type=str)
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--batch-size', type=int, default=1020)
parser.add_argument('--iters', type=int, default=150)
parser.add_argument('--sigma', type=float, default=5.0)
parser.add_argument('--kernel-size', type=int, default=11)
parser.add_argument('--reps', type=int, default=2)
parser.add_argument('--lmbda', type=float, default=0.001)
parser.add_argument('--scale-factor', type=int, default=1)
parser.add_argument('--device',  type=str, default='cuda')
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize(args.size, transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(args.size),
    transforms.ToTensor()
])

data = FlowersDataset(args.data_path, 'test', transform)
loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

# naive inpainting module that uses a Gaussian filter to predict values of masked out pixels
inpainter = Inpainter(args.sigma, args.kernel_size, args.reps, args.scale_factor).to(args.device)
# module that gets mask as input and returns its boundary, used to restrict updates only to boundary pixels
boundary = Boundary().to(args.device)

start_time = time.time()
for batch_idx, (x, seg) in enumerate(loader):
    print(len(x))
    print("Batch {}/{}".format(batch_idx+1, len(loader)))
    x, seg = x.to(args.device), seg.to(args.device)

    # initializes a mask for each sample in the mini batch as a centered square
    mask = torch.nn.Parameter(torch.zeros(len(x), 1, args.size, args.size).to(args.device))
    init_start, init_end = args.size//5, args.size - args.size//5
    mask.data[:,:,init_start:init_end,init_start:init_end].fill_(1.0)

    for i in range(args.iters):
        foreground = x * mask
        background = x * (1-mask)

        pred_foreground = inpainter(background, (1-mask))
        pred_background = inpainter(foreground, mask)

        # inpainting error is equiv to negative coeff. of constraint between foreground and background
        inp_error = neg_coeff_constraint(x, mask, pred_foreground, pred_background)
        # diversity term is the total deviation of foreground and background pixels
        mask_diversity = diversity(x, mask, foreground, background)

        # regularized IEM objective (to be maximized) is the inpainting error minus diversity regularizer
        total_loss = inp_error - args.lmbda * mask_diversity
        total_loss.sum().backward()

        with torch.no_grad():
            grad = mask.grad.data
            # we only update mask pixels that are in the boundary AND have non-zero gradient
            update_bool = boundary(mask) * (grad != 0)
            # pixels with positive gradients are set to 1 and with negative gradients are set to 0
            mask.data[update_bool] = (grad[update_bool] > 0).float()
            grad.zero_()
            
            # smoothing procedure: we set a pixel to 1 if there are 4 or more 1-valued pixels in its 3x3 neighborhood
            mask.data = (F.avg_pool2d(mask, 3, 1, 1, divisor_override=1) >= 4).float()

            acc, iou, miou, dice = compute_performance(mask, seg)
            print("\tIter {:>3}: InpError {:.3f} IoU {:.3f} DICE {:.3f}".format(i, inp_error.mean().item(), iou, dice))

end_time = time.time()
print("IEM finished in {:.1f} seconds".format(end_time-start_time))