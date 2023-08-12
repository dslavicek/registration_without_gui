'''
This code is copied from https://github.com/KrakenLeaf/pytorch_differentiable_mutual_info/blob/master/pytorch_diff_mutual_info.py
Minor adjustments were made, mainly removing parts not relevant for this use case.
'''

#                               Imports                                #
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np


#                               Functions                              #
# ----------------------------------------------------------------------
# Mutual information - numpy implementation for comparison
# Note: This code snippet was taken from the tutorial found at:
#                   https://matthew-brett.github.io/teaching/mutual_information.html
# For comparison purposes
def mutual_information(hgram):
    """Mutual information (MI) for joint histogram
    MI is in fact the Kullback-Leibler (KL) divergence between P_{xy}(x, y) and P_x(x) * P_y(y):

                        I(x, y) = D_{KL}(P_{xy} || p_x * p_y)

    :param hgram: Joint 2D histogram
    :return: MI (scalar)
    """

    # Convert bin counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x

    # Broadcast to multiply marginals. Now we can do the calculation using the pxy, px_py 2D arrays
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum

    # Return the KL-divergence
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


# Entropy - numpy implementation for comparison
def entropy(hgram):
    """
    Calculates the entropy og a given histogram

    :param hgram: Histogram (of any dimension)
    :return: Entropy (scalar)
    """
    px = hgram / float(np.sum(hgram))
    nzs = px > 0  # Only non-zero pxy values contribute to the sum
    x = np.sum(px[nzs] * np.log(px[nzs]))
    return -x


# Note: This code snippet was taken from the discussion found at:
#               https://discuss.pytorch.org/t/differentiable-torch-histc/25865/2
# By Tony-Y
class SoftHistogram1D(nn.Module):
    '''
    Differentiable 1D histogram calculation (supported via pytorch's autograd)
    inupt:
          x     - N x D array, where N is the batch size and D is the length of each data series
          bins  - Number of bins for the histogram
          min   - Scalar min value to be included in the histogram
          max   - Scalar max value to be included in the histogram
          sigma - Scalar smoothing factor fir the bin approximation via sigmoid functions.
                  Larger values correspond to sharper edges, and thus yield a more accurate approximation
    output:
          N x bins array, where each row is a histogram
    '''

    def __init__(self, bins=50, min=0, max=1, sigma=10):
        super(SoftHistogram1D, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)  # Bin centers
        self.centers = nn.Parameter(self.centers, requires_grad=False)  # Wrap for allow for cuda support

    def forward(self, x):
        # Replicate x and for each row remove center
        x = torch.unsqueeze(x, 1) - torch.unsqueeze(self.centers, 1)

        # Bin approximation using a sigmoid function
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))

        # Sum along the non-batch dimensions
        x = x.sum(dim=-1)
        # x = x / x.sum(dim=-1).unsqueeze(1)  # normalization
        return x


# Note: This is an extension to the 2D case of the previous code snippet
class SoftHistogram2D(nn.Module):
    '''
    Differentiable 1D histogram calculation (supported via pytorch's autograd)
    inupt:
          x, y  - N x D array, where N is the batch size and D is the length of each data series
                 (i.e. vectorized image or vectorized 3D volume)
          bins  - Number of bins for the histogram
          min   - Scalar min value to be included in the histogram
          max   - Scalar max value to be included in the histogram
          sigma - Scalar smoothing factor fir the bin approximation via sigmoid functions.
                  Larger values correspond to sharper edges, and thus yield a more accurate approximation
    output:
          N x bins array, where each row is a histogram
    '''

    def __init__(self, bins=50, min=0, max=1, sigma=10):
        super(SoftHistogram2D, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)  # Bin centers
        self.centers = nn.Parameter(self.centers, requires_grad=False)  # Wrap for allow for cuda support

    def forward(self, x, y):
        assert x.size() == y.size(), "(SoftHistogram2D) x and y sizes do not match"

        # Replicate x and for each row remove center
        x = torch.unsqueeze(x, 1) - torch.unsqueeze(self.centers, 1)
        y = torch.unsqueeze(y, 1) - torch.unsqueeze(self.centers, 1)

        # Bin approximation using a sigmoid function (can be sigma_x and sigma_y respectively - same for delta)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        y = torch.sigmoid(self.sigma * (y + self.delta / 2)) - torch.sigmoid(self.sigma * (y - self.delta / 2))

        # Batched matrix multiplication - this way we sum jointly
        z = torch.matmul(x, y.permute((0, 2, 1)))
        return z


class MI_pytorch(nn.Module):
    '''
    This class is a pytorch implementation of the mutual information (MI) calculation between two images.
    This is an approximation, as the images' histograms rely on differentiable approximations of rectangular windows.

            I(X, Y) = H(X) + H(Y) - H(X, Y) = \sum(\sum(p(X, Y) * log(p(Y, Y)/(p(X) * p(Y)))))

    where H(X) = -\sum(p(x) * log(p(x))) is the entropy
    '''

    def __init__(self, bins=50, min=0, max=1, sigma=10, reduction='sum'):
        super(MI_pytorch, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.reduction = reduction

        # 2D joint histogram
        self.hist2d = SoftHistogram2D(bins, min, max, sigma)

        # Epsilon - to avoid log(0)
        self.eps = torch.tensor(0.00000001, dtype=torch.float32, requires_grad=False)

    def forward(self, im1, im2):
        '''
        Forward implementation of a differentiable MI estimator for batched images
        :param im1: N x ... tensor, where N is the batch size
                    ... dimensions can take any form, i.e. 2D images or 3D volumes.
        :param im2: N x ... tensor, where N is the batch size
        :return: N x 1 vector - the approximate MI values between the batched im1 and im2
        '''

        # Check for valid inputs
        assert im1.size() == im2.size(), "(MI_pytorch) Inputs should have the same dimensions."

        batch_size = im1.size()[0]

        # Flatten tensors
        im1_flat = im1.view(im1.size()[0], -1)
        im2_flat = im2.view(im2.size()[0], -1)

        # Calculate joint histogram
        hgram = self.hist2d(im1_flat, im2_flat)

        # Convert to a joint distribution
        # Pxy = torch.distributions.Categorical(probs=hgram).probs
        Pxy = torch.div(hgram, torch.sum(hgram.view(hgram.size()[0], -1)))

        # Calculate the marginal distributions
        Py = torch.sum(Pxy, dim=1).unsqueeze(1)
        Px = torch.sum(Pxy, dim=2).unsqueeze(1)

        # Use the KL divergence distance to calculate the MI
        Px_Py = torch.matmul(Px.permute((0, 2, 1)), Py)

        # Reshape to batch_size X all_the_rest
        Pxy = Pxy.reshape(batch_size, -1)
        Px_Py = Px_Py.reshape(batch_size, -1)

        # Calculate mutual information - this is an approximation due to the histogram calculation and eps,
        # but it can handle batches
        if batch_size == 1:
            # No need for eps approximation in the case of a single batch
            nzs = Pxy > 0  # Calculate based on the non-zero values only
            mut_info = torch.matmul(Pxy[nzs], torch.log(Pxy[nzs]) - torch.log(Px_Py[nzs]))  # MI calculation
        else:
            # For arbitrary batch size > 1
            mut_info = torch.sum(Pxy * (torch.log(Pxy + self.eps) - torch.log(Px_Py + self.eps)), dim=1)

        # Reduction
        if self.reduction == 'sum':
            mut_info = torch.sum(mut_info)
        elif self.reduction == 'batchmean':
            mut_info = torch.sum(mut_info)
            mut_info = mut_info / float(batch_size)

        return mut_info
