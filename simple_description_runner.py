import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL,AutoencoderKLTemporalDecoder,UNetSpatioTemporalConditionModel, EulerDiscreteScheduler
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import os
import numpy as np
import cv2
from collections import Counter
from io import BytesIO
import requests
import tempfile
import moviepy as mp
import lightning as L
from tqdm import tqdm
import tqdm.notebook as tq
from itertools import islice
import gc
from huggingface_hub import login,hf_hub_download
from huggingface_hub import HfApi, HfFolder
import shutil
import traceback
import sys
import re


from typing import List, Tuple, Iterator
from datasets import load_dataset,disable_caching,concatenate_datasets
from datasets.iterable_dataset import _convert_to_arrow,_apply_feature_types_on_example

from datasets.features.features import FeatureType, _align_features, _check_if_features_can_be_aligned, cast_to_python_objects
from datasets.formatting.formatting import PythonFormatter, TensorFormatter
from datasets.formatting import get_format_type_from_alias, get_formatter

sys.path.append('/kaggle/working/VideoCrafter')


## VideoCrafter Specific Imports
from omegaconf import OmegaConf
from functools import partial
from contextlib import contextmanager
import numpy as np
from einops import rearrange, repeat
import logging
mainlogger = logging.getLogger('mainlogger')
from torchvision.utils import make_grid
import pytorch_lightning as pl
from utils.utils import instantiate_from_config
from lvdm.ema import LitEma
from lvdm.distributions import DiagonalGaussianDistribution
from lvdm.models.utils_diffusion import make_beta_schedule
from lvdm.modules.encoders.ip_resampler import ImageProjModel, Resampler
from lvdm.basics import disabled_train
from lvdm.common import (
    extract_into_tensor,
    noise_like,
    exists,
    default
)
from lvdm.models.ddpm3d import DDPM
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps
from lvdm.common import noise_like
from scripts.evaluation.funcs import save_videos



class LatentDiffusion(DDPM):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="caption",
                 cond_stage_trainable=False,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 uncond_prob=0.2,
                 uncond_type="empty_seq",
                 scale_factor=1.0,
                 scale_by_std=False,
                 encoder_type="2d",
                 only_model=False,
                 use_scale=False,
                 scale_a=1,
                 scale_b=0.3,
                 mid_step=400,
                 fix_scale_bug=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        conditioning_key = default(conditioning_key, 'crossattn')
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key

        # scale factor
        self.use_scale=use_scale
        if self.use_scale:
            self.scale_a=scale_a
            self.scale_b=scale_b
            if fix_scale_bug:
                scale_step=self.num_timesteps-mid_step
            else: #bug
                scale_step = self.num_timesteps

            scale_arr1 = np.linspace(scale_a, scale_b, mid_step)
            scale_arr2 = np.full(scale_step, scale_b)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            scale_arr_prev = np.append(scale_a, scale_arr[:-1])
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer('scale_arr', to_torch(scale_arr))

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        # self.instantiate_first_stage(first_stage_config)
        # self.instantiate_cond_stage(cond_stage_config)
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config        
        self.clip_denoised = False

        self.cond_stage_forward = cond_stage_forward
        self.encoder_type = encoder_type
        assert(encoder_type in ["2d", "3d"])
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert(uncond_type in ["zero_embed", "empty_seq"])
        self.uncond_type = uncond_type


        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True
                

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if self.use_scale:  
            return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start *
                extract_into_tensor(self.scale_arr, t, x_start.shape) +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        else:
            return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    def _freeze_model(self):
        for name, para in self.model.diffusion_model.named_parameters():
            para.requires_grad = False

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model
    
    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_first_stage_encoding(self, encoder_posterior, noise=None):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample(noise=noise)
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
   
    @torch.no_grad()
    def encode_first_stage(self, x):
        if self.encoder_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False
        
        encoder_posterior = self.first_stage_model.encode(x)
        results = self.get_first_stage_encoding(encoder_posterior).detach()
        
        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        
        return results
    
    @torch.no_grad()
    def encode_first_stage_2DAE(self, x):

        b, _, t, _, _ = x.shape
        results = torch.cat([self.get_first_stage_encoding(self.first_stage_model.encode(x[:,:,i])).detach().unsqueeze(2) for i in range(t)], dim=2)
        
        return results
    
    def decode_core(self, z, **kwargs):
        if self.encoder_type == "2d" and z.dim() == 5:
            b, _, t, _, _ = z.shape
            z = rearrange(z, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False
            
        z = 1. / self.scale_factor * z

        results = self.first_stage_model.decode(z, **kwargs)
            
        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        return results

    @torch.no_grad()
    def decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    def apply_model(self, x_noisy, t, cond, **kwargs):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond, **kwargs)

        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def _get_denoise_row_from_list(self, samples, desc=''):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_log_timesteps = len(denoise_row)

        denoise_row = torch.stack(denoise_row)  # n_log_timesteps, b, C, H, W
        
        if denoise_row.dim() == 5:
            # img, num_imgs= n_log_timesteps * bs, grid_size=[bs,n_log_timesteps]
            denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
            denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=n_log_timesteps)
        elif denoise_row.dim() == 6:
            # video, grid_size=[n_log_timesteps*bs, t]
            video_length = denoise_row.shape[3]
            denoise_grid = rearrange(denoise_row, 'n b c t h w -> b n c t h w')
            denoise_grid = rearrange(denoise_grid, 'b n c t h w -> (b n) c t h w')
            denoise_grid = rearrange(denoise_grid, 'n c t h w -> (n t) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=video_length)
        else:
            raise ValueError

        return denoise_grid
 

    @torch.no_grad()
    def decode_first_stage_2DAE(self, z, **kwargs):

        b, _, t, _, _ = z.shape
        z = 1. / self.scale_factor * z
        results = torch.cat([self.first_stage_model.decode(z[:,:,i], **kwargs).unsqueeze(2) for i in range(t)], dim=2)

        return results


    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_x0=False, score_corrector=None, corrector_kwargs=None, **kwargs):
        t_in = t
        model_out = self.apply_model(x, t_in, c, **kwargs)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False, return_x0=False, \
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, **kwargs):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised, return_x0=return_x0, \
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, **kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None, verbose=True, callback=None, \
                      timesteps=None, mask=None, x0=None, img_callback=None, start_T=None, log_every_t=None, **kwargs):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]        
        # sample an initial noise
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, **kwargs)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))
        self.use_scale = self.model.use_scale
        print('DDIM scale', self.use_scale)

        if self.use_scale:
            self.register_buffer('scale_arr', to_torch(self.model.scale_arr))
            ddim_scale_arr = self.scale_arr.cpu()[self.ddim_timesteps]
            self.register_buffer('ddim_scale_arr', ddim_scale_arr)
            ddim_scale_arr = np.asarray([self.scale_arr.cpu()[0]] + self.scale_arr.cpu()[self.ddim_timesteps[:-1]].tolist())
            self.register_buffer('ddim_scale_arr_prev', ddim_scale_arr)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps-1,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        
        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=schedule_verbose)
        
        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        
        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    verbose=verbose,
                                                    **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,
                      cond_tau=1., target_size=None, start_timesteps=None,
                      **kwargs):
        device = self.model.betas.device        
        print('ddim device', device)
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
            
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose:
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = time_range

        init_x0 = False
        clean_cond = kwargs.pop("clean_cond", False)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            if start_timesteps is not None:
                assert x0 is not None
                if step > start_timesteps*time_range[0]:
                    continue
                elif not init_x0:
                    img = self.model.q_sample(x0, ts) 
                    init_x0 = True

            # use mask to blend noised original latent (img_orig) & new sampled latent (img)
            if mask is not None:
                assert x0 is not None
                if clean_cond:
                    img_orig = x0
                else:
                    img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass? <ddim inversion>
                img = img_orig * mask + (1. - mask) * img # keep original & modify use img
            
            index_clip =  int((1 - cond_tau) * total_steps)
            if index <= index_clip and target_size is not None:
                target_size_ = [target_size[0], target_size[1]//8, target_size[2]//8]
                img = torch.nn.functional.interpolate(
                img,
                size=target_size_,
                mode="nearest",
                )
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      x0=x0,
                                      **kwargs)
            
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      uc_type=None, conditional_guidance_scale_temporal=None, **kwargs):
        b, *_, device = *x.shape, x.device
        if x.dim() == 5:
            is_video = True
        else:
            is_video = False
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, **kwargs) # unet denoiser
        else:
            # with unconditional condition
            if isinstance(c, torch.Tensor):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            elif isinstance(c, dict):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            else:
                raise NotImplementedError
            # text cfg
            if uc_type is None:
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            else:
                if uc_type == 'cfg_original':
                    e_t = e_t + unconditional_guidance_scale * (e_t - e_t_uncond)
                elif uc_type == 'cfg_ours':
                    e_t = e_t + unconditional_guidance_scale * (e_t_uncond - e_t)
                else:
                    raise NotImplementedError
            # temporal guidance
            if conditional_guidance_scale_temporal is not None:
                e_t_temporal = self.model.apply_model(x, t, c, **kwargs)
                e_t_image = self.model.apply_model(x, t, c, no_temporal_attn=True, **kwargs)
                e_t = e_t + conditional_guidance_scale_temporal * (e_t_temporal - e_t_image)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        
        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t

        with open('/kaggle/working/Imagine-Gen-Inference-UI/debug3.txt','a') as f:
            f.write(f'sigma_t {sigma_t} sigmas {sigmas[index]} e_t {e_t}')
        
        #if index==0:
        #    sigma_t = torch.full(size, 0, device=device)
        
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        with open('/kaggle/working/Imagine-Gen-Inference-UI/debug2.txt','a') as f:
            f.write(f'a_prev {a_prev} dir_xt {dir_xt} noise {noise}')
        if self.use_scale:
            scale_arr = self.model.scale_arr if use_original_steps else self.ddim_scale_arr
            scale_t = torch.full(size, scale_arr[index], device=device)
            scale_arr_prev = self.model.scale_arr_prev if use_original_steps else self.ddim_scale_arr_prev
            scale_t_prev = torch.full(size, scale_arr_prev[index], device=device)
            pred_x0 /= scale_t 
            x_prev = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
        else:
            print('no scale')
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        with open('/kaggle/working/Imagine-Gen-Inference-UI/debug.txt','a') as f:
            f.write(f'x_prev {x_prev} pred_x0 {pred_x0}')
        return x_prev, pred_x0


    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)

        def extract_into_tensor(a, t, x_shape):
            b, *_ = t.shape
            out = a.gather(-1, t)
            return out.reshape(b, *((1,) * (len(x_shape) - 1)))

        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec



# Constants
NUM_PROC_DS_PROCESSING = 20
MAX_WORD_COUNT_LIMIT = 77 #Prev 80
BATCH_SIZE = 16
TRAIN_BATCH_SIZE = 8
AUTOENC_IMG_HEIGHT = 256
AUTOENC_IMG_WIDTH = 256
KEYFRAME_COUNT = 16
DATALOADER_NUM_PROCS = 6
DTYPE_PT = torch.float32
EMBEDDING_DIMENSION = 4096
CROSS_ATTN_TEXT_CONTEXT_LEN = MAX_WORD_COUNT_LIMIT

with open('/kaggle/working/VideoCrafter/.env','w') as f:
    env_str=f'CROSS_ATTN_TEXT_CONTEXT_LEN = {CROSS_ATTN_TEXT_CONTEXT_LEN}'
    f.write(env_str)


#Bridge Autoencoder Class
class BridgeAutoencoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.embedding_dim = EMBEDDING_DIMENSION
        
        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
        )
        
        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, self.embedding_dim),
            nn.Identity()
        )
        
    def forward(self, x):
        # Encode and decode
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        # Forward pass and compute loss
        x = batch
        x_reconstructed = self(x)
        loss = nn.HuberLoss()(x_reconstructed, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # Optimizer
        return torch.optim.Adam(self.parameters())



# PyTorch LightningModule Class
class InteractiveChatVideoGenModel(L.LightningModule):
    def __init__(
            self, 
            vae="maxin-cn/Latte-1", 
            vae_subfolder="vae_temporal_decoder", 
            cross_attention_dim=1024, 
            multimodal_llm="llava-hf/LLaVA-NeXT-Video-7B-hf", 
            train_timesteps=1000,
            unet_model_channels=192,
            unet_use_gradient_checkpointing=False,
            llm_low_mem_cpu_usage=True, 
            height=256, 
            width=256,
            num_frames=16,
            noise_aug_strength=0.02, 
            do_classifier_free_guidance=True, 
            cfg_uncond_prob=0.1,
            basic_lr=0.0001, 
            weight_decay=0.0, 
            betas=(0.9, 0.999), 
            batch_size=8,
            use_llm=True, 
            use_tpu_vm=False, 
            use_vae=False,
            tpu_mesh=None, 
            fsdp_auto_wrap_policy=None, 
            shard_output_callback=None, 
            use_xla=False,
            yaml_config_file_path='/kaggle/working/VideoCrafter/configs/inference_t2v_512_v2.0.yaml'):
            
        super(InteractiveChatVideoGenModel, self).__init__()
        self.height = height
        self.width = width
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.noise_aug_strength = noise_aug_strength
        
        self.vae = None
        self.vae_id = vae
        self.vae_subfolder = vae_subfolder
        self.latent_size = 32
        self.latent_channels = 4
        self.train_timesteps = train_timesteps
        self.num_frames = num_frames
        self.cfg_uncond_prob = cfg_uncond_prob
        self.yaml_config_file_path = yaml_config_file_path
        
        
        self.unet = None
        self.latent_diffusion_model = None
        self.ddim_sampler = None
        self.cross_attention_dim = cross_attention_dim
        
        self.multimodal_llm = None
        self.multimodal_llm_id = multimodal_llm
        self.llm_low_mem_cpu_usage = llm_low_mem_cpu_usage
        self.multimodal_llm_processor = None
        # self.noise_scheduler = DDIMScheduler()
        # self.noise_scheduler = EulerDiscreteScheduler()
        
        self.vae_scale_factor = None
        self.basic_lr = basic_lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.batch_size = batch_size
        self.use_llm = use_llm
        self.use_tpu_vm = use_tpu_vm
        self.use_xla = use_xla
        self.use_vae = use_vae
        # self.bridge_mlp = None
        self.bridge_autoencoder = None
        self.tpu_mesh=tpu_mesh
        
        config = OmegaConf.load(self.yaml_config_file_path)
        model_config = config.pop("model", OmegaConf.create())
        
        self.latent_diffusion_model_config = model_config
        self.unet_model_channels = unet_model_channels
        self.unet_use_gradient_checkpointing = unet_use_gradient_checkpointing
        
        self.latent_diffusion_model_config['params']['unet_config']['params']['model_channels'] = self.unet_model_channels
        
        self.latent_diffusion_model_config['params']['unet_config']['params']['use_checkpoint'] = self.unet_use_gradient_checkpointing
        
        self.latent_diffusion_model_config['params']['image_size'] = [self.latent_size,self.latent_size]
        
        self.latent_diffusion_model_config['params']['unet_config']['params']['temporal_length'] = self.num_frames
        
        self.latent_diffusion_model_config['params']['uncond_type']  = "zero_embed"
        if self.use_tpu_vm:
            self.tpu_mesh = tpu_mesh
            self.fsdp_auto_wrap_policy = fsdp_auto_wrap_policy
            self.shard_output_callback = shard_output_callback

    # Model Configuration Methods

    def configure_model(self):
        if self.multimodal_llm is None and self.use_llm:
            self.multimodal_llm = LlavaNextVideoForConditionalGeneration.from_pretrained(
                self.multimodal_llm_id, low_cpu_mem_usage=self.llm_low_mem_cpu_usage, torch_dtype=DTYPE_PT)
            if self.use_tpu_vm:
                self.multimodal_llm = SpmdFullyShardedDataParallel(
                    self.multimodal_llm, mesh=self.tpu_mesh, auto_wrap_policy=self.fsdp_auto_wrap_policy, shard_output=self.shard_output_callback)
            self.multimodal_llm.config.text_config.output_hidden_states = True
            self.multimodal_llm.eval()
            for param in self.multimodal_llm.parameters():
                param.requires_grad = False

        if self.multimodal_llm_processor is None and self.use_llm:
            self.multimodal_llm_processor = LlavaNextVideoProcessor.from_pretrained(
                self.multimodal_llm_id)

        if self.vae is None and self.use_vae:
            self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
                self.vae_id, subfolder=self.vae_subfolder)
            if self.use_tpu_vm:
                self.vae = SpmdFullyShardedDataParallel(
                    self.vae, mesh=self.tpu_mesh, auto_wrap_policy=self.fsdp_auto_wrap_policy, shard_output=self.shard_output_callback)

            self.vae_scale_factor = 2 ** (
                len(self.vae.config.block_out_channels)-1)
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
                
                
        if self.unet is None:
            self.latent_diffusion_model = LatentDiffusion(**self.latent_diffusion_model_config.get("params", dict()))
            self.unet = self.latent_diffusion_model.model.diffusion_model
            self.ddim_sampler = DDIMSampler(self.latent_diffusion_model)
            #self.ddim_sampler.make_schedule(self.train_timesteps,ddim_eta=1,verbose=False)

        # if self.bridge_mlp is None:
        #     self.bridge_mlp = nn.Sequential(
        #         nn.Linear(4096, 2048), nn.Tanh(), nn.Linear(2048, 1024))

        if self.bridge_autoencoder is None:
            self.bridge_autoencoder = BridgeAutoencoder()
            self.bridge_autoencoder.eval()
            for param in self.bridge_autoencoder.parameters():
                param.requires_grad = False

        # if self.unetspatiotemporalconditionmodel is None:
        #     self.unetspatiotemporalconditionmodel = UNetSpatioTemporalConditionModel(sample_size=(self.latent_size, self.latent_size), cross_attention_dim=self.cross_attention_dim, in_channels=self.latent_channels,
        #                                                                             out_channels=self.latent_channels, down_block_types=self.down_block_types, up_block_types=self.up_block_types, block_out_channels=self.block_out_channels, num_attention_heads=self.num_attention_heads)
        #     # if DTYPE_PT==torch.float16:
        #     #    self.unetspatiotemporalconditionmodel = self.unetspatiotemporalconditionmodel.half()
        #     if self.use_tpu_vm:
        #         self.unetspatiotemporalconditionmodel = SpmdFullyShardedDataParallel(
        #             self.unetspatiotemporalconditionmodel, mesh=self.tpu_mesh, auto_wrap_policy=self.fsdp_auto_wrap_policy, shard_output=self.shard_output_callback)

    # XLA Specific Methods

    def xla_mark_step(self):
        xm.mark_step()

    # VAE Methods
    @torch.no_grad()
    def vae_encode(self, tensor):
        return self.vae.encode(tensor)

    @torch.no_grad()
    def vae_decode(self, z):
        return self.vae.decode(z)

    @torch.no_grad()
    def vae_temp_decode(self,z,num_frames=None):
        if num_frames is None:
            num_frames=self.num_frames
        return self.vae.decode(z,num_frames=num_frames)

    @torch.no_grad()
    def get_vae_latent_from_frame(self, frame):
        encoded_img = self.vae_encode(frame)
        z = encoded_img.latent_dist.sample()
        z = z*self.vae.config.scaling_factor
        return z
    
    @torch.no_grad()
    def get_decoded_frame_from_latent(self, z):
        return self.vae_decode(z).sample

    @torch.no_grad()
    def get_temp_decoded_frame_from_latent(self,z,num_frames=None):
        return self.vae_temp_decode(z,num_frames=num_frames).sample

    def encode_vae_image(self, image, do_classifier_free_guidance):
        image_latent = get_vae_latent_from_frame(image)
        if do_classifier_free_guidance:
            negative_image_latent = torch.zeros_like(image_latent)
            image_latent = torch.cat([negative_image_latent, image_latent])
        return image_latent

    def retrieve_timestamps(self, num_inference_steps, timesteps=None):
        if timesteps is not None:
            self.noise_scheduler.set_timesteps(timesteps=timesteps)
            timesteps = self.noise_scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.noise_scheduler.set_timesteps(num_inference_steps)
            timesteps = self.noise_scheduler.timesteps

        return timesteps, num_inference_steps

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, generator, latents):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, dtype=dtype, device=self.unetspatiotemporalconditionmodel.device)
        else:
            latents = latents

        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents, num_frames, decode_chunk_size=8):
        latents = latents.flatten(0, 1)

        latents = 1/self.vae.config.scaling_factor * latents

        frames = []

        for i in range(0, latents.shape[0], decode_chunk_size):
            frame = self.get_decoded_frame_from_latent(
                latents[i:i+decode_chunk_size])
            frames.append(frame)

        frames = torch.cat(frames, dim=0)

        frames = frames.reshape(-1, num_frames, *
                                frames.shape[1:]).permute(0, 2, 1, 3, 4)

        return frames

    # Time Id methods
    def get_add_time_ids(self, fps, motion_bucket_id, noise_aug_strength, batch_size, do_classifier_free_guidance, train=False):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unetspatiotemporalconditionmodel.config.addition_time_embed_dim * \
            len(add_time_ids)
        expected_add_embed_dim = self.unetspatiotemporalconditionmodel.add_embedding.linear_1.in_features

        add_time_ids = torch.tensor([add_time_ids], device=self.unetspatiotemporalconditionmodel.device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def tpu_mem_usage(self):
        for i in range(8):
            pass
            # print(f"TPU:{i+1}",torch_xla.core.xla_model.get_memory_info(f"xla:{i}") )

    # Multimodal LLM Methods
    def get_prompt_from_conversation(self, conversation_list):
        prompt = self.multimodal_llm_processor.apply_chat_template(
            conversation_list, add_generation_prompt=True)
        return prompt

    def process_prompt(self, prompt):
        # inputs=processor(text=prompt,padding=True, return_tensors="pt").to(self.multimodal_llm.device)
        inputs = self.multimodal_llm_processor(
            text=prompt, padding=True, return_tensors="pt")
        return inputs

    def get_llm_output(self, inputs, max_new_tokens=250):
        output = self.multimodal_llm.generate(
            **inputs, return_dict_in_generate=True, output_hidden_states=True, max_new_tokens=max_new_tokens, do_sample=False)
        return output

    def get_llm_response_text(self, output_sequences):
        llm_response_text = self.multimodal_llm_processor.decode(
            output_sequences, skip_special_tokens=True)
        return llm_response_text

    def get_llm_output_sequences_from_llm_output(self, llm_output):
        return llm_output.sequences[0]

    def get_last_layer_hidden_states_from_llm_output(self, llm_output):
        return torch.cat([item[-1].unsqueeze(0) for item in llm_output.hidden_states[1:]], dim=0)

    def get_last_layer_hidden_states_from_llm_output_batched(self, b_llm_output):
        output_hidden_states = [row[-1]
                                for row in b_llm_output.hidden_states[1:]]
        output_hidden_states_reshaped = []
        for i in range(self.batch_size):
            curr_conv = []
            for j in range(len(output_hidden_states)):
                curr_conv.append(output_hidden_states[j][i])
            curr_conv = torch.concatenate(curr_conv, dim=0)
            output_hidden_states_reshaped.append(curr_conv)
        output_hidden_states_reshaped = torch.stack(
            output_hidden_states_reshaped)
        return output_hidden_states_reshaped

    def get_last_layer_hidden_states_from_llm_output_batched_torch(self,b_llm_output):
        output_hidden_states = [row[-1] for row in b_llm_output.hidden_states[1:]]
        output_hidden_states  = [tensor.to("cpu") for tensor in output_hidden_states] ## why to cpu??
        t_output = [torch.stack(list(pair)) for pair in zip(*output_hidden_states)]
        return t_output

    # Text Processing Methods
    def extract_text_after_token(self, text, token="ASSISTANT:"):
        pattern = re.compile(
            f'{re.escape(token)}(.*?)(?=$|ASSISTANT:)', re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            last_match = matches[-1].strip()
            return last_match
        else:
            return ''

    def get_model_response_appended_conversation(self, conversation, llm_response_text):
        current_text = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": llm_response_text},
            ],
        }
        conversation.append(current_text)
        return conversation

    def get_next_conversation_prompt_from_conversation(self, conversation, text_description, initial_prompt=False):
        if text_description is None:
            text_description = ''
        if initial_prompt:
            current_text = {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"Understand what is being conveyed in the given prompt. Your task is to form one simple sentence less than eight words to explain this to a toddler. Do not include more than one adjective in your response. Do not repeat any word. Your response must be less than eight words. Prompt:'{text_description}'."
                    },
                ],
            }
        else:
            current_text = {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"Understand what is being conveyed in the given prompt. Your task is to form one simple sentence less than eight words to explain this to a toddler. Do not include more than one adjective in your response. Do not repeat any word. Your response must be less than eight words. Prompt:'{text_description}'."
                    },
                ],
            }
        conversation.append(current_text)
        prompt = self.get_prompt_from_conversation(conversation)
        return conversation, prompt

    def _append_dims(self, x, target_dims):
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
        return x[(...,) + (None,) * dims_to_append]


    # Bridge Autoencoder methods
    @torch.no_grad()
    def bridge_ae_encode(self, hidden_states):
        return self.bridge_autoencoder.encoder(hidden_states)

    @torch.no_grad()
    def bridge_ae_decode(self, downsampled_hidden_states):
        return self.bridge_autoencoder.decoder(hidden_states)

    
    @torch.no_grad()
    def sample_latents(self,encoder_hidden_states,num_inference_steps=50,guidance_scale=3,seed=None,return_all_latents=False,disable_progress_bar=False,scheduler=None,fps=8,motion_bucket_id=127,noise_aug_strength=0.02):
        if scheduler is None:
            scheduler = self.noise_scheduler
            scheduler.set_timesteps(num_inference_steps)
            
        rnd_generator = torch.Generator().manual_seed(seed) if seed is not None else None
        
        latents = torch.randn((self.batch_size,self.num_frames,self.latent_channels,self.latent_size,self.latent_size),generator=rnd_generator)
        
        latents = latents.to(self.unetspatiotemporalconditionmodel.device)
        
        latents = latents*scheduler.init_noise_sigma
        
        all_latents = []
        
        guidance_scale_new = torch.linspace(1.0, 3.0, self.num_frames,device=self.unetspatiotemporalconditionmodel.device).unsqueeze(0)
        guidance_scale_new = guidance_scale_new.repeat(self.batch_size, 1)
        guidance_scale_new = self._append_dims(guidance_scale_new, latents.ndim)

        added_time_ids=self.get_add_time_ids(fps,motion_bucket_id,noise_aug_strength,self.batch_size,self.do_classifier_free_guidance,train=False)
        
        if self.do_classifier_free_guidance:
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states,dtype=encoder_hidden_states.dtype,device=self.unetspatiotemporalconditionmodel.device)
            encoder_hidden_states = torch.cat([uncond_encoder_hidden_states,encoder_hidden_states],dim=0)

        for sample_idx,t in enumerate(tqdm(scheduler.timesteps)):
            latent_model_input = torch.cat([latents]*2) if self.do_classifier_free_guidance else latents
            #print(f"{latent_model_input.shape}")
            
            scheduler.scale_model_input(latent_model_input)
            noise_pred = self.unetspatiotemporalconditionmodel(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                added_time_ids=added_time_ids,
            ).sample
            
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            if sample_idx == len(scheduler.timesteps) - 1:
                # compute denoised sample at x_0
                latents = scheduler.step(noise_pred, t, latents).pred_original_sample
                # latents = scheduler.step(noise_pred, t, latents).prev_sample
            else:
                # compute the previous noisy sample at x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # save the latents
            if return_all_latents:
                all_latents.append(latents)
                
        if return_all_latents:
            return torch.cat(all_latents, dim=0)
        
        return latents

    
    
    
    
    @torch.no_grad()
    def remove_noise(
        self,
        noisy_samples,
        noise,
        timesteps,
        scale=1.0,
    ):
        

        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=noisy_samples.device, dtype=noisy_samples.dtype)
        timesteps = timesteps.to(noisy_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        # print(f'noisy_samples DIM :{noisy_samples.shape}\tsqrt_one_minus_alpha_prod DIM:{sqrt_one_minus_alpha_prod.shape}\tnoise DIM:{noise.shape}')
        original_samples = (
            noisy_samples - sqrt_one_minus_alpha_prod * noise) / (sqrt_alpha_prod * scale)
        return original_samples

    def forward(self, input_hidden_states, height=256, width=256, num_frames=16, num_inference_steps=50, min_guidance_scale=1.0, max_guidance_scale=3.0, fps=8, motion_bucket_id=127, noise_aug_strength=0.02, decode_chunk_size=8, generator=None, latents=None, tmax=None,curr_batch_size=None):
        self.guidance_scale = max_guidance_scale

        if curr_batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = curr_batch_size

        # Get Time Added IDs
        # Args: fps,motion_bucket_id,noise_aug_strength,batch_size,do_classifier_free_guidance
        added_time_ids = self.get_add_time_ids(
            fps, motion_bucket_id, noise_aug_strength, batch_size, self.do_classifier_free_guidance, train=True)

        # Prepare timesteps
        # timesteps,num_inference_steps = self.retrieve_timesteps(num_inference_steps)
        if tmax is None:
            tmax = self.noise_scheduler.config.num_train_timesteps

        num_channels_latents = self.unetspatiotemporalconditionmodel.config.in_channels
        # Prepare latent variables
        # Args: batch_size,num_frames,num_channels_latents,height,width,dtype,generator,latents
        latents = self.prepare_latents(batch_size, num_frames, num_channels_latents,
                                       height, width, input_hidden_states.dtype, generator, latents)

        # Prepare guidance scale
        # guidance_scale = torch.linspace(
        #     min_guidance_scale, max_guidance_scale, num_frames, device=self.unetspatiotemporalconditionmodel.device).unsqueeze(0)
        # guidance_scale = guidance_scale.repeat(batch_size, 1)
        # guidance_scale = self._append_dims(guidance_scale, latents.ndim)

        timestep_batch_size = 2*latents.shape[0] if self.do_classifier_free_guidance else latents.shape[0]

        # timestep_batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, tmax, (timestep_batch_size,),
            dtype=torch.long,
            device=self.unetspatiotemporalconditionmodel.device
        ) 
        
        # timesteps = self.noise_scheduler

        num_warmup_steps = len(timesteps) - \
            num_inference_steps * self.noise_scheduler.order
        self._num_timesteps = len(timesteps)

        latents = torch.cat([torch.zeros_like(latents,dtype=latents.dtype,device=self.unetspatiotemporalconditionmodel.device),latents]) if self.do_classifier_free_guidance else latents

        noise = torch.randn_like(latents, device=self.unetspatiotemporalconditionmodel.device)

        noise = self.noise_aug_strength*noise
        
        noisy_latents = self.noise_scheduler.add_noise(
            latents, noise, timesteps)

        latent_model_input = noisy_latents

        if self.do_classifier_free_guidance:
           uncond_input_hidden_states = torch.zeros_like(input_hidden_states,dtype=input_hidden_states.dtype,device=self.unetspatiotemporalconditionmodel.device)
           input_hidden_states = torch.cat([uncond_input_hidden_states,input_hidden_states],dim=0)

        if self.tpu_mesh is not None:
            xs.mark_sharding(latent_model_input,self.tpu_mesh,('data',None,None,None,None)) # (batch_size,16,16,32,32)
            xs.mark_sharding(timesteps,self.tpu_mesh,('data',))
            xs.mark_sharding(input_hidden_states,self.tpu_mesh,('data',None,None)) #(batch,seq_len,4096)
            xs.mark_sharding(added_time_ids,self.tpu_mesh,('data',None))
        
        noise_pred = self.unetspatiotemporalconditionmodel(
            latent_model_input,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            added_time_ids=added_time_ids,
        ).sample

        if self.do_classifier_free_guidance:
           noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
           noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        if self.do_classifier_free_guidance:
            noisy_latents = noisy_latents.chunk(2)[1]
        
        if self.do_classifier_free_guidance:
            timesteps = timesteps.chunk(2)[1]
            
        if self.tpu_mesh is not None:
            xs.mark_sharding(noisy_latents,self.tpu_mesh,('data',None,None,None,None)) # (batch,16,16,32,32)
            xs.mark_sharding(noise_pred,self.tpu_mesh,('data',None,None,None,None))
            xs.mark_sharding(timesteps,self.tpu_mesh,('data',))
        
        pred_latents = self.remove_noise(
            noisy_latents.detach(), noise_pred.detach(), timesteps)

        pred_latents = torch.clamp(pred_latents, latents.min(), latents.max())

        if self.do_classifier_free_guidance:
            noise = noise.chunk(2)[1]
        
        if self.tpu_mesh is not None:
            xs.mark_sharding(noise,self.tpu_mesh,('data',None,None,None,None))
            xs.mark_sharding(pred_latents,self.tpu_mesh,('data',None,None,None,None))
        
        # frames = self.decode_latents(latents,num_frames, decode_chunk_size)

        return pred_latents, noise_pred, noise
        
    def train_unet(self,model_input,context,curr_batch_size=None):
        if curr_batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = curr_batch_size
        
        timesteps = torch.randint(
            0, self.train_timesteps, (batch_size,),
            dtype=torch.long,
            device=model_input.device
        )
        
        noise = torch.randn_like(model_input, device=model_input.device)
        
        noisy_model_input = self.ddim_sampler.stochastic_encode(model_input,timesteps,noise=noise)
        
        if self.do_classifier_free_guidance:
            if torch.rand(1).item() < self.cfg_uncond_prob:
                context = torch.zeros_like(context,dtype=context.dtype,device=model_input.device)
        
        if self.tpu_mesh is not None:
            xs.mark_sharding(noise,self.tpu_mesh,('data',None,None,None,None))
            xs.mark_sharding(model_input,self.tpu_mesh,('data',None,None,None,None)) # (batch_size,4,16,32,32)
            xs.mark_sharding(noisy_model_input,self.tpu_mesh,('data',None,None,None,None)) # (batch_size,4,16,32,32)
            xs.mark_sharding(timesteps,self.tpu_mesh,('data',))
            xs.mark_sharding(context,self.tpu_mesh,('data',None,None)) #(batch,seq_len,1024) with bridge_mlp
        
        noise_pred = self.unet(noisy_model_input,timesteps,context,x0=None,temporal_length=self.num_frames)
        
        if self.tpu_mesh is not None:
            xs.mark_sharding(noise,self.tpu_mesh,('data',None,None,None,None))
            xs.mark_sharding(noise_pred,self.tpu_mesh,('data',None,None,None,None))
        
        
        return noise_pred,noise
        
        

    def training_step(self, batch, batch_idx):

        b_video_frames, b_llm_hidden_states = batch
 
        curr_batch_size = b_video_frames.shape[0]
        
        b_video_frames = b_video_frames.flatten(0, 1)
        
        if self.tpu_mesh is not None:
            xs.mark_sharding(b_video_frames,self.tpu_mesh,('data',None,None,None))
        

        b_video_frame_latents = self.get_vae_latent_from_frame(b_video_frames)
        
        del b_video_frames

        b_video_frame_latents = b_video_frame_latents.view(curr_batch_size, int(
            b_video_frame_latents.shape[0]/curr_batch_size), *b_video_frame_latents.shape[1:])

        curr_batch_size = b_llm_hidden_states.shape[0]
        
        b_llm_hidden_states = b_llm_hidden_states.flatten(0, 1)
        
        if self.tpu_mesh is not None:
            xs.mark_sharding(b_llm_hidden_states,self.tpu_mesh,('data',None))

        # bridge_net_op = self.bridge_mlp(b_llm_hidden_states)

        bridge_net_op = self.bridge_autoencoder.encoder(b_llm_hidden_states)

        bridge_net_op = bridge_net_op.view(curr_batch_size, int(
            bridge_net_op.shape[0]/curr_batch_size), *bridge_net_op.shape[1:])
            
        b_video_frame_latents = rearrange(b_video_frame_latents,'b t c h w -> b c t h w') # Very IMP VideoCrafter UNet expects in this format

        # print(f'{b_video_frame_latents.shape}')
        
        noise_pred,noise = self.train_unet(b_video_frame_latents,bridge_net_op,curr_batch_size=curr_batch_size)
        
        loss = F.mse_loss(noise_pred, noise)

        self.log(f"Training Loss", loss.detach())
        
        return loss

    def validation_step(self, batch, batch_idx):
        b_video_frames, b_llm_hidden_states = batch
 
        curr_batch_size = b_video_frames.shape[0]
        
        b_video_frames = b_video_frames.flatten(0, 1)
        
        if self.tpu_mesh is not None:
            xs.mark_sharding(b_video_frames,self.tpu_mesh,('data',None,None,None))
        

        b_video_frame_latents = self.get_vae_latent_from_frame(b_video_frames)
        
        del b_video_frames

        b_video_frame_latents = b_video_frame_latents.view(curr_batch_size, int(
            b_video_frame_latents.shape[0]/curr_batch_size), *b_video_frame_latents.shape[1:])

        curr_batch_size = b_llm_hidden_states.shape[0]
        
        b_llm_hidden_states = b_llm_hidden_states.flatten(0, 1)
        
        if self.tpu_mesh is not None:
            xs.mark_sharding(b_llm_hidden_states,self.tpu_mesh,('data',None))

        # bridge_net_op = self.bridge_mlp(b_llm_hidden_states)

        bridge_net_op = self.bridge_autoencoder.encoder(b_llm_hidden_states)

        bridge_net_op = bridge_net_op.view(curr_batch_size, int(
            bridge_net_op.shape[0]/curr_batch_size), *bridge_net_op.shape[1:])
        
        b_video_frame_latents = rearrange(b_video_frame_latents,'b t c h w -> b c t h w')

        noise_pred,noise = self.train_unet(b_video_frame_latents,bridge_net_op,curr_batch_size=curr_batch_size)
        
        loss = F.mse_loss(noise_pred, noise)

        self.log(f"Validation Loss", loss.detach())
        
        return loss

    def configure_optimizers(self):
        #params=list(self.unetspatiotemporalconditionmodel.parameters())+list(self.bridge_mlp.parameters())
        # params = list(self.unet.parameters())+list(self.bridge_mlp.parameters())
        params = list(self.unet.parameters())
        return torch.optim.AdamW(params, lr=self.basic_lr, weight_decay=self.weight_decay, betas=self.betas)


def batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, **kwargs):
    ddim_sampler = model.ddim_sampler
    #ddim_sampler  = DDIMSampler(model.latent_diffusion_model)
    uncond_type = model.latent_diffusion_model.uncond_type
    batch_size = noise_shape[0]

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.latent_diffusion_model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.latent_diffusion_model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    x_T = None
    batch_variants = []
    #batch_variants1, batch_variants2 = [], []
    for _ in range(n_samples):
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=True,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            x_T=x_T,
                                            **kwargs
                                            )
        ## reconstruct from latent to pixel space
        z = samples
        b, c, t, h, w = z.shape
        z = 1. / model.vae.config.scaling_factor * z #(-19,20)-> (-4,4)
        #print(z.shape,z[:,:,0].shape)
        #sys.exit()
        z = rearrange(z,'b c t h w -> b t c h w')
        z = z.flatten(0,1)
        frames = model.get_temp_decoded_frame_from_latent(z)
        frames = rearrange(frames,'t c h w -> 1 c t h w')
        #results = torch.cat([model.get_temp_decoded_frame_from_latent(z[:,:,i]).unsqueeze(2) for i in range(t)], dim=2)

        #batch_images = model.decode_first_stage_2DAE(samples)
        batch_variants.append(frames)
    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants


def load_model(model_repo_id,model_file_name):
    global simple_int_gen_model
    """ Load the UNET """
    login("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")
    checkpoint_path = hf_hub_download(repo_id=model_repo_id, filename=model_file_name)

    print(checkpoint_path)
        
    state_dict = torch.load(checkpoint_path)

    simple_int_gen_model = InteractiveChatVideoGenModel(
    batch_size=TRAIN_BATCH_SIZE, use_llm=False, use_tpu_vm=False, use_xla=False,use_vae=True)

    simple_int_gen_model.configure_model()

    simple_int_gen_model.load_state_dict(state_dict)

    #print(simple_int_gen_model.latent_diffusion_model.alphas_cumprod)

    simple_int_gen_model.latent_diffusion_model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=simple_int_gen_model.latent_diffusion_model_config['params']['timesteps'],linear_start=simple_int_gen_model.latent_diffusion_model_config['params']['linear_start'], linear_end=simple_int_gen_model.latent_diffusion_model_config['params']['linear_end'], cosine_s=8e-3)
    
    simple_int_gen_model.multimodal_llm=LlavaNextVideoForConditionalGeneration.from_pretrained(simple_int_gen_model.multimodal_llm_id,torch_dtype=DTYPE_PT,load_in_4bit=True,device_map="cuda:0")

    simple_int_gen_model.multimodal_llm_processor = LlavaNextVideoProcessor.from_pretrained(simple_int_gen_model.multimodal_llm_id)


simple_int_gen_model = None

@torch.no_grad()
def run_simple_desc_inference(prompt, model_repo_id,model_file_name,num_inference_steps=50,fps=24,ddim_eta=1.0,unconditional_guidance_scale=1.0):
    
    global simple_int_gen_model
    
    if simple_int_gen_model is None:
        print("First run, loading model...")
        load_model(model_repo_id,model_file_name)

    ### UNCOMMENT THE BELOW BLOCK UPTO INDICATION
    b_text_description=[prompt]

    b_llm_text_input = [simple_int_gen_model.get_next_conversation_prompt_from_conversation([],text_desc,initial_prompt=True) for text_desc in b_text_description]

    b_llm_input_processed = simple_int_gen_model.multimodal_llm_processor([elem[1] for elem in b_llm_text_input],padding=True, return_tensors="pt")

    b_llm_output = simple_int_gen_model.multimodal_llm.generate(b_llm_input_processed['input_ids'],return_dict_in_generate=True,output_hidden_states=True, max_new_tokens=MAX_WORD_COUNT_LIMIT, do_sample=False)

    b_llm_hidden_states =  simple_int_gen_model.get_last_layer_hidden_states_from_llm_output_batched_torch(b_llm_output)[0]
    #print("Raw HS shape: ", b_llm_hidden_states.shape)
    b_llm_hidden_states=b_llm_hidden_states.flatten(0,1)
    #print("Flattened HS shape: ", b_llm_hidden_states.shape)
    simple_int_gen_model.latent_diffusion_model=simple_int_gen_model.latent_diffusion_model.to("cuda")
    
    simple_int_gen_model.bridge_autoencoder=simple_int_gen_model.bridge_autoencoder.to("cuda")

    simple_int_gen_model.vae=simple_int_gen_model.vae.to("cuda")

    b_llm_hidden_states=b_llm_hidden_states.to("cuda")
    
    # bridge_net_op=simple_int_gen_model.bridge_mlp(b_llm_hidden_states)

    bridge_net_op=simple_int_gen_model.bridge_ae_encode(b_llm_hidden_states)
    #print("Encoded bridge op shape: ", bridge_net_op.shape)

    bridge_net_op=bridge_net_op.unsqueeze(0)
    #print("Unsqueezed bridge op shape: ", bridge_net_op.shape)

    ### UNCOMMENT UPTO THE LINE ABOVE THIS

    #print(f'{bridge_net_op.shape}')

    #sys.exit()


    ### TESTING BY DIRECTLY GETTING BRIDGE_AE OPS FROM REPO

    # b_text_description = [prompt]
    # b_llm_text_input = [int_chat_video_gen_model.get_next_conversation_prompt_from_conversation([],text_desc,initial_prompt=True) for text_desc in b_text_description]

    # ds = load_dataset("amrithagk/capstone_bridge_ae_outputs", data_files="seg1_part1/train-00000-of-00004.parquet", split='train')

    # #tgt_sample = ds.filter(lambda example: example['llm_op_text']==b_llm_text_input[0])

    # #bridge_net_op = tgt_sample['bridge_net_op']
    # print("desc", ds[1]['text_description'])
    # bridge_net_op = torch.tensor(ds[1]['bridge_net_op'])[:11, :]
    # print("Bridge op shape:", bridge_net_op.shape)

    # bridge_net_op=torch.tensor(bridge_net_op).unsqueeze(0)
    # print("Unsqueezed bridge op shape: ", bridge_net_op.shape)
    # print("Got bridge net ops")

    ### COMMENT UPTO ABOVE LINE TO PERFORM ON-THE-GO LLM OPS
    

    simple_int_gen_model.batch_size=1

    simple_int_gen_model.eval()

    noise_shape = [simple_int_gen_model.batch_size, simple_int_gen_model.latent_diffusion_model.channels, simple_int_gen_model.num_frames, simple_int_gen_model.latent_size, simple_int_gen_model.latent_size]
    print("noise dims generated")    
    cond = {"c_crossattn": [bridge_net_op], "fps": fps}

    batch_samples = batch_ddim_sampling(simple_int_gen_model, cond, noise_shape, 1, \
                                                num_inference_steps, ddim_eta, unconditional_guidance_scale)
    print("got batch samples")    
    save_videos(batch_samples, '/kaggle/working/Imagine-Gen-Inference-UI/outputs/', [f'{prompt[:10]}_is_{num_inference_steps}_cfg_{unconditional_guidance_scale}_eta_{ddim_eta}_{prompt}'], fps=8)
    print("Saved video")
    return "/kaggle/working/Imagine-Gen-Inference-UI/outputs/" + f'{prompt[:10]}_is_{num_inference_steps}_cfg_{unconditional_guidance_scale}_eta_{ddim_eta}_{prompt}' + ".mp4"
    
if __name__=="__main__":
    #text_prompt="A majestic fantasy castle with a grand fountain and ornate architecture under a blue sky, in a city with ocean views."
    text_prompt_1 = 'A monochrome comic scene captures a yellow snowflake falling from a house roof, with a simple black background and focus on text.'
    text_prompt_2 = 'Large indoor amusement park ceiling filled with intricate gears and futuristic scenery, reflecting in windows with close-up of robot chains at night.'
    #text_prompt_3 = "A smiling boy"
    model_repo_id="amrithagk/capstone_model"
    model_file_name="pytorch_model_unet_cross_attn_192_tmp_dcd_bridge_ae_simple_desc_ddim_optical_flow_eph_1_sakuga.bin"
    #num_inference_steps=500
    fps=8
    ddim_eta=1.0
    #unconditional_guidance_scale=8.0

    l_num_inference_steps = [50,100,200,500]
    l_unconditional_guidance_scale = [1.0,4.0,8.0]
    # l_num_inference_steps = [200]
    # l_unconditional_guidance_scale = [8.0]

    ## Download and configure our model
    checkpoint_path = hf_hub_download(repo_id=model_repo_id, filename=model_file_name)

    print(checkpoint_path)

    state_dict = torch.load(checkpoint_path)

    int_chat_video_gen_model = InteractiveChatVideoGenModel(
    batch_size=TRAIN_BATCH_SIZE, use_llm=False, use_tpu_vm=False, use_xla=False,use_vae=True)

    int_chat_video_gen_model.configure_model()

    int_chat_video_gen_model.load_state_dict(state_dict)

    #print(int_chat_video_gen_model.latent_diffusion_model.alphas_cumprod)

    int_chat_video_gen_model.latent_diffusion_model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=int_chat_video_gen_model.latent_diffusion_model_config['params']['timesteps'],linear_start=int_chat_video_gen_model.latent_diffusion_model_config['params']['linear_start'], linear_end=int_chat_video_gen_model.latent_diffusion_model_config['params']['linear_end'], cosine_s=8e-3)

    int_chat_video_gen_model.multimodal_llm=LlavaNextVideoForConditionalGeneration.from_pretrained(int_chat_video_gen_model.multimodal_llm_id,torch_dtype=DTYPE_PT,load_in_4bit=True,device_map="cuda:0")

    int_chat_video_gen_model.multimodal_llm_processor = LlavaNextVideoProcessor.from_pretrained(int_chat_video_gen_model.multimodal_llm_id)

    
    # Run inference loop
    for i,i_steps in enumerate(l_num_inference_steps):
        for j,cfg_scale in enumerate(l_unconditional_guidance_scale):
            
            run_simple_desc_inference(text_prompt_2, int_chat_video_gen_model, model_repo_id,model_file_name,i_steps,fps,ddim_eta,cfg_scale)
    
    # run_inference(text_prompt, model_repo_id,model_file_name,num_inference_steps,fps,ddim_eta,unconditional_guidance_scale)