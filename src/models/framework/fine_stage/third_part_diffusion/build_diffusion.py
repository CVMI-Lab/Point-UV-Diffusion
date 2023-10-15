"""
The code for diffusion prior in DALLE2, refered from https://github.com/lucidrains/DALLE2-pytorch
"""

import math
import pdb
import random
import time

from tqdm.auto import tqdm
from functools import partial, wraps
from contextlib import contextmanager
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum
import torchvision.transforms as T

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many, check_shape
from einops_exts.torch import EinopsToAndFrom
import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# sys.path.append(os.path.dirname(__file__))
# from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def l2norm(t):
    return F.normalize(t, dim=-1)


def exists(val):
    return val is not None


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def extract(a, t, x_shape):
    a = a.to(t.device) 
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / first(alphas_cumprod)
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def quadratic_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps, dtype=torch.float64) ** 2


def sigmoid_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(-6, 6, timesteps, dtype=torch.float64)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class DiffusionPrior(nn.Module):
    def __init__(
            self,
            net,
            *,
            loss_fn: nn.Module,
            clip=None,
            image_embed_dim=None,
            image_channels=3,
            timesteps=1000,
            cond_drop_prob=0.,
            predict_x_start=True,
            beta_schedule="cosine",
            condition_on_text_encodings=True,
            input_scaler=False, # refer from http://arxiv.org/abs/2301.10972
            # the paper suggests this is needed, but you can turn it off for your CLIP preprocessed text embed -> image embed training
            sampling_clamp_l2norm=False,
            training_clamp_l2norm=False,
            init_image_embed_l2norm=False,
            image_embed_scale=None,
            # this is for scaling the l2-normed image embedding, so it is more suitable for gaussian diffusion, as outlined by Katherine (@crowsonkb) https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132
            clip_adapter_overrides=dict()
    ):
        super().__init__()
        self.input_scaler = input_scaler
        self.noise_scheduler = NoiseScheduler(
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            loss_fn=loss_fn,
            input_scaler=input_scaler,
        )

        if exists(clip):
            assert image_channels == clip.image_channels, f'channels of image ({image_channels}) should be equal to the channels that CLIP accepts ({clip.image_channels})'

            if isinstance(clip, CLIP):
                clip = XClipAdapter(clip, **clip_adapter_overrides)
            elif isinstance(clip, CoCa):
                clip = CoCaAdapter(clip, **clip_adapter_overrides)

            assert isinstance(clip, BaseClipAdapter)
            freeze_model_and_make_eval_(clip)
            self.clip = clip
        else:
            assert exists(
                image_embed_dim), 'latent dimension must be given, if training prior network without CLIP given'
            self.clip = None

        self.net = net
        self.image_embed_dim = default(image_embed_dim, lambda: clip.dim_latent)

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.
        self.condition_on_text_encodings = condition_on_text_encodings

        # in paper, they do not predict the noise, but predict x0 directly for image embedding, claiming empirically better results. I'll just offer both.
        self.predict_x_start = predict_x_start

        # @crowsonkb 's suggestion - https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132
        self.image_embed_scale = default(image_embed_scale, self.image_embed_dim ** 0.5)

        # whether to force an l2norm, similar to clipping denoised, when sampling
        self.sampling_clamp_l2norm = sampling_clamp_l2norm
        self.training_clamp_l2norm = training_clamp_l2norm
        self.init_image_embed_l2norm = init_image_embed_l2norm

        # device tracker
        self.register_buffer('_dummy', torch.tensor([True]), persistent=False)

        self.timesteps = timesteps
        self.beta_schedule = beta_schedule

    @property
    def device(self):
        return self._dummy.device

    def p_mean_variance(self, x, t, cond, self_cond=None, clip_denoised=False, cond_scale=1.):
        assert not (
                    cond_scale != 1. and not self.can_classifier_guidance), 'the model was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        if cond == None:
            pred = self.net(
                x,
                t,
            )
        else:
            pred = self.net(
                x,
                t,
                cond,
            )

        # pred = self.net.forward_with_cond_scale(x, t, cond_scale = cond_scale, **text_cond)

        if self.predict_x_start:
            x_start = pred
            # not 100% sure of this above line - for any spectators, let me know in the github issues (or through a pull request) if you know how to correctly do this
            # i'll be rereading https://arxiv.org/abs/2111.14822, where i think a similar approach is taken
        else:
            x_start = self.noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        #if clip_denoised and not self.predict_x_start:
        if not self.predict_x_start:
            '''
            # dynamic threshold
            s = torch.quantile(
                rearrange(x_start, 'b ... -> b (...)').abs(),
                0.95,
                dim=-1
            )

            s.clamp_(min=3.)
            s = right_pad_dims_to(x_start, s)
            x_start = x_start.clamp(-s, s) / s * 3
            '''
            x_start.clamp_(-1., 1.)

        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_start = l2norm(x_start) * self.image_embed_scale

        model_mean, posterior_variance, posterior_log_variance = self.noise_scheduler.q_posterior(x_start=x_start,
                                                                                                  x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, cond=None, self_cond=None, clip_denoised=True, cond_scale=1.):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=t, cond=cond, self_cond=None,
                                                                          clip_denoised=clip_denoised,
                                                                          cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, device, shape, cond=None, clip_denoised=True, cond_scale=1., cond_drop_time=False):
        # device = self.device

        b = shape[0]
        image_embed = torch.randn(shape, device=device)
        x_start = None  # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step',
                      total=self.noise_scheduler.num_timesteps):
            times = torch.full((b,), i, device=device, dtype=torch.long)
            # self_cond = x_start if self.net.self_cond else None
            self_cond = None

            if self.cond_drop_prob != 0:
                if i/self.noise_scheduler.num_timesteps <= cond_drop_time:
                    if isinstance(cond, dict):
                        cond['other_cond'][:, :3, ...] = 0
                    else:
                        cond[:, :3, ...] = 0


            if self.input_scaler:
                image_embed = image_embed/image_embed.std(axis=(1, 2, 3), keepdims=True)
            image_embed, x_start = self.p_sample(image_embed, times, cond=cond, clip_denoised=clip_denoised,
                                                 self_cond=self_cond, cond_scale=cond_scale)

        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddim(self, device, shape, timesteps, eta=1., cond=None, clip_denoised=True, cond_scale=1., cond_drop_time=False):
        batch, _, alphas, total_timesteps = shape[
            0], self.device, self.noise_scheduler.alphas_cumprod_prev, self.noise_scheduler.num_timesteps

        times = torch.linspace(-1., total_timesteps, steps=timesteps + 1)[:-1]

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        b = shape[0]
        image_embed = torch.randn(shape, device=device)

        x_start = None  # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            alpha = alphas[time]
            alpha_next = alphas[time_next]

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            if self.cond_drop_prob != 0:
                if time/self.noise_scheduler.num_timesteps <= cond_drop_time:
                    if isinstance(cond, dict):
                        cond['other_cond'][:, :3, ...] = 0
                    else:
                        cond[:, :3, ...] = 0


            # self_cond = x_start if self.net.self_cond else None
            if self.input_scaler:
                image_embed = image_embed/image_embed.std(axis=(1, 2, 3), keepdims=True)
            if cond == None:
                pred = self.net(
                    image_embed,
                    time_cond,
                )
            else:
                pred = self.net(
                    image_embed,
                    time_cond,
                    cond,
                )

            # derive x0

            if self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(image_embed, t=time_cond, noise=pred)

            # clip x0 before maybe predicting noise

            # if not self.predict_x_start:
            #     x_start.clamp_(-1., 1.)
            #
            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)

            # predict noise

            pred_noise = self.noise_scheduler.predict_noise_from_start(image_embed, t=time_cond, x0=x_start)

            if time_next < 0:
                image_embed = x_start
                continue

            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
            noise = torch.randn_like(image_embed) if time_next > 0 else 0.

            image_embed = x_start * alpha_next.sqrt() + \
                          c1 * noise + \
                          c2 * pred_noise

        # if self.predict_x_start and self.sampling_final_clamp_l2norm:
        #     image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddpm(self, device, shape, cond=None, clip_denoised=True, cond_scale=1., cond_drop_time=False):

        b = shape[0]
        image_embed = torch.randn(shape, device = device)
        x_start = None

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps):
            times = torch.full((batch,), i, device = device, dtype = torch.long)

            self_cond = x_start if self.net.self_cond else None
            if self.input_scaler:
                image_embed = image_embed/image_embed.std(axis=(1, 2, 3), keepdims=True)
            image_embed, x_start = self.p_sample(image_embed, times, cond=cond, clip_denoised=clip_denoised,
                                                 self_cond=self_cond, cond_scale=cond_scale)

        return image_embed

    def p_losses(self, image_embed, times, cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(image_embed))
        image_embed_noisy = self.noise_scheduler.q_sample(x_start=image_embed, t=times, noise=noise)
        if cond == None:
            pred = self.net(
                image_embed_noisy,
                times,
            )
        else:
            if self.cond_drop_prob != 0:
                if random.uniform(0, 1) < self.cond_drop_prob:
                    if isinstance(cond, dict):
                        cond['other_cond'][:, :3, ...] = 0
                    else:
                        cond[:, :3, ...] = 0

            pred = self.net(
                image_embed_noisy,
                times,
                cond,
            )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = l2norm(pred) * self.image_embed_scale
        target = noise if not self.predict_x_start else image_embed
        loss = self.noise_scheduler.loss_fn(pred, target)

        if self.predict_x_start:
            pred_start = pred
        else:
            pred_start = self.noise_scheduler.predict_start_from_noise(image_embed_noisy, t=times, noise=pred)

        return loss, pred_start, image_embed_noisy

    @torch.no_grad()
    @eval_decorator
    def sample_batch_size(self, batch_size, text_cond, cond_scale=1.):
        device = self.betas.device
        shape = (batch_size, self.image_embed_dim)

        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step',
                      total=self.noise_scheduler.num_timesteps):
            img = self.p_sample(img, torch.full((batch_size,), i, device=device, dtype=torch.long), text_cond=text_cond,
                                cond_scale=cond_scale)
        return img

    @torch.no_grad()
    @eval_decorator
    def sample(self, text, num_samples_per_batch=2, cond_scale=1.):
        # in the paper, what they did was
        # sample 2 image embeddings, choose the top 1 similarity, as judged by CLIP
        text = repeat(text, 'b ... -> (b r) ...', r=num_samples_per_batch)

        batch_size = text.shape[0]
        image_embed_dim = self.image_embed_dim

        text_embed, text_encodings, text_mask = self.clip.embed_text(text)

        text_cond = dict(text_embed=text_embed)

        if self.condition_on_text_encodings:
            text_cond = {**text_cond, 'text_encodings': text_encodings, 'mask': text_mask}

        image_embeds = self.p_sample_loop((batch_size, image_embed_dim), text_cond=text_cond, cond_scale=cond_scale)

        # retrieve original unscaled image embed

        image_embeds /= self.image_embed_scale

        text_embeds = text_cond['text_embed']

        text_embeds = rearrange(text_embeds, '(b r) d -> b r d', r=num_samples_per_batch)
        image_embeds = rearrange(image_embeds, '(b r) d -> b r d', r=num_samples_per_batch)

        text_image_sims = einsum('b r d, b r d -> b r', l2norm(text_embeds), l2norm(image_embeds))
        top_sim_indices = text_image_sims.topk(k=1).indices

        top_sim_indices = repeat(top_sim_indices, 'b 1 -> b 1 d', d=image_embed_dim)

        top_image_embeds = image_embeds.gather(1, top_sim_indices)
        return rearrange(top_image_embeds, 'b 1 d -> b d')

    def forward(
            self,
            image_embed=None,
            cond=None,
            *args,
            **kwargs
    ):

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = torch.randint(0, self.noise_scheduler.num_timesteps, (batch,), device=device, dtype=torch.long)

        # scale image embed (Katherine)
        # image_embed *= self.image_embed_scale
        # import pdb;pdb.set_trace()
        # calculate forward loss

        return self.p_losses(image_embed, times, cond=cond, *args, **kwargs)



def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


class NoiseScheduler(nn.Module):
    def __init__(self, *, beta_schedule, timesteps, loss_fn, input_scaler=False, p2_loss_weight_gamma=0., p2_loss_weight_k=1):
        super().__init__()

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == "jsd":
            betas = 1.0 / torch.linspace(timesteps, 1, timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise NotImplementedError()

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.loss_fn = loss_fn
        self.input_scaler = input_scaler

        # register buffer helper function to cast double back to float

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # p2 loss reweighting

        self.has_p2_loss_reweighting = p2_loss_weight_gamma > 0.
        register_buffer('p2_loss_weight',
                        (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def sample_random_times(self, batch):
        return torch.randint(0, self.num_timesteps, (batch,), device=self.betas.device, dtype=torch.long)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.input_scaler:
            x_start = x_start*self.input_scaler
            x_t = (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
            x_t = x_t/x_t.std(axis=(1, 2, 3), keepdims=True)
        else:
            x_t = (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        return x_t

    def q_sample_from_to(self, x_from, from_t, to_t, noise=None):
        shape = x_from.shape
        noise = default(noise, lambda: torch.randn_like(x_from))

        alpha = extract(self.sqrt_alphas_cumprod, from_t, shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, from_t, shape)
        alpha_next = extract(self.sqrt_alphas_cumprod, to_t, shape)
        sigma_next = extract(self.sqrt_one_minus_alphas_cumprod, to_t, shape)

        return x_from * (alpha_next / alpha) + noise * (sigma_next * alpha - sigma * alpha_next) / alpha

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def p2_reweigh_loss(self, loss, times):
        if not self.has_p2_loss_reweighting:
            return loss
        return loss * extract(self.p2_loss_weight, times, loss.shape)