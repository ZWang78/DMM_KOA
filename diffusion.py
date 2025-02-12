import torch
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Sinusoidal position embedding for timesteps.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
import math
import torch
import torch.nn.functional as F
import numpy as np

def linear_beta_schedule(timesteps):
    """
    Linearly increase beta from 0.0001 to 0.02, scaled by timesteps.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule, from https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion:
    """
    Core diffusion process class.
    """
    def __init__(self, timesteps=100, beta_schedule='linear', loss_ncc=None, loss_reg=None):
        self.timesteps = timesteps

        # Choose beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        self.betas = betas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # diff terms
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # Registration losses (NCC + smoothness) from transModel
        self.loss_ncc = loss_ncc
        self.loss_reg = loss_reg

    def _extract(self, a, t, x_shape):
        """
        Extract values from a for batch indices t, reshaping to x_shape.
        """
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0) = sqrt_bar_alpha_t x_0 + sqrt(1 - bar_alpha_t)*noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_mean_variance(self, x_start, t):
        """
        Mean and variance of q(x_t | x_0).
        """
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Reverse of q_sample (given x_t and predicted noise, guess x_0).
        """
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        """
        Model predicts noise, from which we derive x_0. Then get mean, var of q-posterior.
        """
        pred_noise = model(x_t, t)
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        """
        Sample from p(x_{t-1} | x_t).
        """
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1]*(len(x_t.shape)-1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, model, x_start, x_T, continous=True):
        """
        A placeholder function for your specific usage in shape transform.
        Returns model's code, plus placeholders for demonstration.
        """
        import numpy as np

        S = x_start
        T = x_T
        x_0 = x_T
        b, c, h, w = S.shape

        t = torch.full((b,), 0, device=S.device, dtype=torch.long)
        # Model expects cat of [S, T, x_noisy], but we do a direct call for demonstration:
        score = model(torch.cat([S, T, x_0], dim=1), t)

        # Demo placeholders
        gamma = np.linspace(0, 1, 5)
        flow_stack = torch.zeros([b, 2, h, w], device=S.device)
        code_stack = score
        defm_stack = S

        return score, code_stack, defm_stack, flow_stack

    @torch.no_grad()
    def p_sample_loop_validation_0_4(self, model, x_start, score, continous=True):
        """
        Another placeholder function for validation usage.
        """
        import numpy as np
        S = x_start
        b, c, h, w = S.shape

        t = torch.full((b,), 0, device=S.device, dtype=torch.long)
        score_ave = score

        gamma = np.linspace(0, 1, 5)
        flow_stack = torch.zeros([b, 2, h, w], device=S.device)
        code_stack = score_ave
        defm_stack = S

        return code_stack, defm_stack, flow_stack

    @torch.no_grad()
    def sample(self, model, x_start, x_T, continous=True):
        return self.p_sample_loop(model, x_start, x_T, continous)

    @torch.no_grad()
    def sample_validation(self, model, x_start, score, continous=True):
        return self.p_sample_loop_validation_0_4(model, x_start, score, continous)

    def train_losses(self, model, trans, x_start, x_T, t):
        """
        Compute training loss as the sum of:
        - MSE loss on predicted noise
        - Registration losses: ncc + gradient (flow)
        """
        noise = torch.randn_like(x_T)
        x_noisy = self.q_sample(x_T, t, noise=noise)

        # predict noise from x_noisy
        predicted_noise = model(torch.cat([x_start, x_T, x_noisy], dim=1), t)
        loss_mse = torch.nn.functional.mse_loss(noise, predicted_noise)

        # pass [x_start, predicted_noise] into transform block
        output, flow = trans(torch.cat([x_start, predicted_noise], dim=1))

        l_sim = self.loss_ncc(output, x_T)
        l_smt = self.loss_reg(flow)

        loss = loss_mse + (l_sim + l_smt) * 50
        return loss, output, x_T, predicted_noise, x_start, flow, x_noisy
