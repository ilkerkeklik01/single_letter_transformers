import torch


def cache(length, dim, theta, device):
    frequency = 1.0 / theta ** (torch.arange(0, dim, 2, device=device).float() / dim)
    angle = torch.outer(torch.arange(length, device=device).float(), frequency)
    return angle.cos()[None, None], angle.sin()[None, None]


def apply(x, cos, sin):
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)
