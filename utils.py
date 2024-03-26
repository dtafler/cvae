import torch
import torch.nn.functional as F
from torch import nn, Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterator, Callable, Sequence, Tuple
from torch.utils.data import Sampler, TensorDataset, Dataset
from torch.distributions import MultivariateNormal, Distribution


class Linear_Classifier(nn.Module):
    def __init__(self, num_feats, num_classes):
        super().__init__()
        self.lin = nn.Linear(num_feats, num_classes)
        
    def forward(self, x):
        return self.lin(x)
    
class N_Layer_Dense_Classifier(nn.Module):
    def __init__(self, num_feats, num_classes, num_layers, dropout_rate=0.0):
        super().__init__()
        layers = []
        for _ in range(num_layers -1):
            layers.extend([
                nn.Linear(num_feats, num_feats), 
                nn.ReLU(), 
                nn.Dropout(dropout_rate)
            ])
        layers.append(nn.Linear(num_feats, num_classes))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    
def accuracy(out, y):
    preds = torch.argmax(out, dim=1)
    return (preds == y).float().mean()


def average(metric: tuple, nums: tuple):
    return sum(torch.tensor(metric) * torch.tensor(nums)) / sum(torch.tensor(nums))


def loss_batch(X, y, model, loss_fn, opt=None, acc=False):
    preds = model(X)
    loss = loss_fn(preds, y)
      
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    if acc:
        return loss.item(), len(X), accuracy(preds, y)
    
    else:
        return loss.item(), len(X)
    
    
def fit(epochs, model, loss_fn, opt, train_dl, test_dl, device="cpu"):
    model.to(device)
    
    train_loss_epoch = []
    train_acc_epoch = []
    test_loss_epoch = []
    test_acc_epoch = []
    
    for epoch in range(epochs):
        
        model.train()
        losses, nums, accs = zip(*[loss_batch(X.to(device),y.to(device), model, loss_fn, opt, acc=True) for X,y in tqdm(train_dl, desc=f'Epoch {epoch + 1} training')])
        train_loss = average(losses, nums)
        train_acc = average(accs, nums) 
            
        model.eval()
        with torch.no_grad():
            losses, nums, accs = zip(*[loss_batch(X.to(device), y.to(device), model, loss_fn, acc=True) for X, y in tqdm(test_dl, desc=f'Epoch {epoch + 1} testing')])

        test_loss = average(losses, nums)
        test_acc = average(accs, nums)            
        
        # print(f"Average train loss: {train_loss:.4f}")
        # print(f"Average train accuracy: {train_acc:.4f}")
        # print(f"Average test loss: {test_loss:.4f}")
        # print(f"Average test accuracy: {test_acc:.4f}\n")
        
        train_loss_epoch.append(train_loss)
        train_acc_epoch.append(train_acc)
        test_loss_epoch.append(test_loss)
        test_acc_epoch.append(test_acc)
    return train_loss_epoch, train_acc_epoch, test_loss_epoch, test_acc_epoch


def plot_train_val(train_y, test_y, ylabel, extrema_fn=min):
    epochs = len(train_y)
    fig, ax = plt.subplots()
    ax.plot(range(epochs), train_y, label='train')
    ax.plot(range(epochs), test_y, label='test')
    ax.legend()
    ax.set_xticks(range(epochs))
    ax.set_xlabel('epochs')
    ax.set_ylabel(ylabel)
    
    extr_train = extrema_fn(train_y)
    epoch_train = train_y.index(extr_train)
    extr_test = extrema_fn(test_y)
    epoch_test = test_y.index(extr_test)

    ax.plot(epoch_train, extr_train, 'ro')
    ax.plot(epoch_test, extr_test, 'ro')
    
    ax.axvline(x=epoch_train, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=epoch_test, color='gray', linestyle='--', alpha=0.5)

    ax.annotate(f"{extr_train:.4f}", (epoch_train, extr_train), textcoords="offset points", xytext=(-10,-10), ha='center')
    ax.annotate(f"{extr_test:.4f}", (epoch_test, extr_test), textcoords="offset points", xytext=(-10,-10), ha='center')

    return fig


# TODO: License? adapted from here: https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification/blob/34a61e432881816c2da14d577d6ed63501288f5f/loss/BalancedSoftmaxLoss.py#L37C1-L51C16
def balanced_softmax_loss(logits, labels, sample_per_class, reduction='mean'):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


# TODO: rewrite with torch
def samples_per_class(labels: np.ndarray):
    return torch.tensor([(l == labels).sum() for l in np.unique(labels)])

  
class TensorDatasetTransform(Dataset[Tuple[Tensor, Tensor]]):
    r"""Dataset wrapping tensors and optionally transforming samples.
    
    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        tensors (Tuple): two tensors (features, labels) that have the same size of the first dimension.
        tranform (Callable | None): function that transforms each sample of first tensor.
        transform_probs (Tensor): probabilities of transformation for all samples.
    """

    def __init__(self, tensors: Tuple[Tensor, Tensor], 
                 transform: Callable[[Tensor], Tensor] | None = None,
                 transform_probs: Tensor | None = None) -> None:
        
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.transform = transform
        self.trans_probs = transform_probs
        
        if self.transform:
            assert self.trans_probs is not None, "Transform_probs not provided, although transform was provided."
        
    def __getitem__(self, index):
        feat, label = self.tensors[0][index], self.tensors[1][index]
        if self.transform:
            if torch.bernoulli(self.trans_probs[index]):
                feat = self.transform(feat)
        return feat, label

    def __len__(self):
        return self.tensors[0].size(0)
    
    
def add_noise(tensor: Tensor, std: float) -> Tensor : 
    noise = torch.normal(0, 0.5, size=tensor.shape)
    # print(abs(noise).mean())
    return tensor + noise

def n_largest_avg_per_cls_stats(feats: Tensor, labels: Tensor, n_largest: int) -> Tuple[Tensor, Tensor]:
    """
    Calculate the mean and average (across classes) standard deviation of the embeddings for the n largest classes.

    Args:
        feats (Tensor): The input feature tensor.
        labels (Tensor): The corresponding label tensor.
        n_largest (int): The number of largest classes to consider.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the mean and average standard deviation tensors.
    """
    labels = labels.numpy()
    feats = feats.numpy()
    # split into classes
    class_embs = {}
    for l in set(labels):
        idx = np.asarray(labels == l).nonzero()[0]
        class_embs[l] = feats[idx]

    # get n_largest largest classes
    largest_cls_idx = sorted(class_embs.keys(), key=lambda x: class_embs[x].shape[0], reverse=True)[:n_largest]
    largest_cls_embs = np.concatenate([class_embs[idx] for idx in largest_cls_idx], axis=0)

    # calculate statistics
    mean = np.mean(largest_cls_embs, axis=0)
    stds = np.concatenate([np.expand_dims(np.std(class_embs[idx], axis=0), axis=0) for idx in largest_cls_idx], axis=0)
    mean_std = np.mean(stds, axis=0)
    
    return torch.Tensor(mean), torch.Tensor(mean_std)


# def add_variation(tensor: Tensor, std: Tensor, mult: float) -> Tensor:
#     """
#     Adds variation to a tensor by adding gaussian noise with given standard deviation tensor.

#     Args:
#         tensor (Tensor): The input tensor.
#         std (Tensor): The standard deviation tensor.
#         mult (float): The multiplier for the standard deviation.

#     Returns:
#         Tensor: The tensor with added variation.
#     """
#     assert tensor.ndim == 1, "Tensor has more than one dimension."
#     assert std.shape == tensor.shape, "Shapes of tensor and std don't match."
    
#     # noise = np.random.multivariate_normal(np.zeros(tensor.shape), mult * np.diag(std)**2)
#     # noise = Tensor(noise)
#     noise = MultivariateNormal(torch.zeros(tensor.shape), mult * torch.diag(std**2)).sample()
#     # print(abs(noise).mean())
#     return tensor + noise

def add_variation(tensor: Tensor, distribution: Distribution, mult: float = 1) -> Tensor:
    noise = distribution.sample() * mult
    # print(abs(noise).mean())
    return tensor + noise
    


class ClassBalancedSampler(Sampler[int]):
    def __init__(self, data: TensorDataset | TensorDatasetTransform) -> None:
        assert isinstance(data, TensorDataset) or isinstance(data, TensorDatasetTransform), 'data not of type TensorDataset'
        self.labels = data.tensors[1]
        self.classes = self.labels.unique()
        self.class_size = self.labels.bincount().max()

    def __iter__(self) -> Iterator[int]:
        class_idx = {cl.item():(self.labels == cl).nonzero(as_tuple=True)[0] for cl in self.classes}
        idx = torch.zeros(len(self)).to(int)
        for i, cl in enumerate(self.classes):
            inds = class_idx[cl.item()]
            mult = self.class_size // len(inds)
            final_inds = torch.cat([inds for _ in range(mult)])
            rest = inds[torch.randperm(len(inds))][:self.class_size - len(final_inds)]
            final_inds = torch.cat((final_inds, rest))
            idx[i*self.class_size : (i+1)*self.class_size] = final_inds 
        yield from idx[torch.randperm(len(idx))]
    
    def __len__(self) -> int:
        return self.class_size * len(self.classes)
