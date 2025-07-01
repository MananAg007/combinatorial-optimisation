"""Schedulers for Denoising Diffusion Probabilistic Models"""

import math

import numpy as np
import torch


class GaussianDiffusion(object):
  """Gaussian Diffusion process with linear beta scheduling"""

  def __init__(self, T, schedule):
    # Diffusion steps
    self.T = T

    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    self.betabar = np.cumprod(self.beta)
    self.alpha = np.concatenate((np.array([1.0]), 1 - self.beta))
    self.alphabar = np.cumprod(self.alpha)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0, t):
    # Select noise scales
    noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
    atbar = torch.from_numpy(self.alphabar[t]).view(noise_dims).to(x0.device)
    assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'

    # Sample noise and add to x0
    epsilon = torch.randn_like(x0)
    xt = torch.sqrt(atbar) * x0 + torch.sqrt(1.0 - atbar) * epsilon
    return xt, epsilon


class CategoricalDiffusion(object):
  """Categorical Diffusion process with linear beta scheduling"""

  def __init__(self, T, schedule):
    # Diffusion steps
    self.T = T

    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    beta = self.beta.reshape((-1, 1, 1))
    eye = np.eye(2).reshape((1, 2, 2))
    ones = np.ones((2, 2)).reshape((1, 2, 2))

    self.Qs = (1 - beta) * eye + (beta / 2) * ones

    Q_bar = [np.eye(2)]
    for Q in self.Qs:
      Q_bar.append(Q_bar[-1] @ Q)
    self.Q_bar = np.stack(Q_bar, axis=0)

    self.Q_bar_torch = torch.from_numpy(self.Q_bar).float()
    self.Q_bar_inv_torch = torch.linalg.inv(self.Q_bar_torch)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0_onehot, t):
    # Select noise scales
    Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)
    xt = torch.matmul(x0_onehot, Q_bar.reshape((Q_bar.shape[0], 1, 2, 2)))
    return torch.bernoulli(xt[..., 1].clamp(0, 1))


class DiffusionForcing:
  """Helper class to implement token-wise denoising schedules from diffusion-forcing paper."""
  
  def __init__(self, T):
    """Initialize the diffusion forcing scheduler.
    
    Args:
      T: The total number of diffusion steps
    """
    self.T = T
    
  def generate_pyramid_scheduling_matrix(self, horizon, uncertainty_scale):
    """Generate a pyramid scheduling matrix for token-wise denoising.
    
    This creates a schedule that progressively denoises tokens from left to right
    with an uncertainty scale that determines how many timesteps to wait
    before denoising the next token.
    
    Args:
      horizon: The number of tokens to denoise
      uncertainty_scale: Higher values mean more tokens are denoised jointly
                         Lower values approach autoregressive denoising
    
    Returns:
      A scheduling matrix of shape (height, horizon) where each row represents
      a timestep and each column represents a token
    """
    height = self.T + int((horizon - 1) * uncertainty_scale) + 1
    scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
    for m in range(height):
      for t in range(horizon):
        scheduling_matrix[m, t] = self.T + int(t * uncertainty_scale) - m
    
    return np.clip(scheduling_matrix, 0, self.T)
  
  def generate_autoregressive_scheduling_matrix(self, horizon):
    """Generate a scheduling matrix for fully autoregressive denoising.
    
    This is equivalent to a pyramid schedule with uncertainty_scale = T
    
    Args:
      horizon: The number of tokens to denoise
    
    Returns:
      A scheduling matrix for autoregressive denoising
    """
    return self.generate_pyramid_scheduling_matrix(horizon, self.T)
  
  def generate_full_sequence_scheduling_matrix(self, horizon):
    """Generate a scheduling matrix for full sequence denoising.
    
    All tokens are denoised simultaneously in this schedule.
    
    Args:
      horizon: The number of tokens to denoise
    
    Returns:
      A scheduling matrix for full sequence denoising
    """
    return np.arange(self.T, -1, -1)[:, None].repeat(horizon, axis=1)


class InferenceSchedule(object):
  def __init__(self, inference_schedule="linear", T=1000, inference_T=1000):
    self.inference_schedule = inference_schedule
    self.T = T
    self.inference_T = inference_T

  def __call__(self, i):
    assert 0 <= i < self.inference_T

    if self.inference_schedule == "linear":
      t1 = self.T - int((float(i) / self.inference_T) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    elif self.inference_schedule == "cosine":
      t1 = self.T - int(
          np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int(
          np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    else:
      raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))
