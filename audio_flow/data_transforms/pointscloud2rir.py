import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from audio_flow.encoders.bigvgan import Mel_BigVGAN_44kHz


class PointsCloud2RIR(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.vocoder = Mel_BigVGAN_44kHz()

    def __call__(self, data: dict) -> tuple[Tensor, dict, dict]:
        r"""Transform data into latent representations and conditions.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: mel bins
        """
        
        name = data["dataset_name"][0]
        device = next(self.parameters()).device

        if name in ["FDTD_2D_RIR_PointsCloud"]:
                # "bnd_pointscloud": bnd_pointscloud[None, ...], # (1, l_bnd, 2)
                # "u0": u0[None, ...], # (1, h, w)
                # "x": x[None, ...], # (1, 1)
                # "y": y[None, ...], # (1, 1)
                # "uxy": uxy[None, ...], # (1, l, 1, 1)
        
            u0 = data["u0"].to(device) # (b, 1, t, f)
            bnd_pointscloud = data["bnd_pointscloud"].to(device) # (b, 1, l_bnd, 2)
            x = data["x"].to(device) # (b, 1,)
            y = data["y"].to(device) # (b, 1,)
            uxy = data["uxy"].to(device)  # (b, l, 1, 1)
            target = rearrange(uxy, 'b l t f -> b t l f')  # (b, 1, l, 1)
            l1 = 240
            target = rearrange(target, 'b c (l1 l2) f -> b c l1 (l2 f)', l1=l1)  # (b, 1, 240, l2)
            
            cond_c = torch.cat([x, y], dim=1) # (b, 2,)
            cond_nn = u0 # (b, 1, 240, l2)
            cond_avg = bnd_pointscloud # (b, 1, l, 2)

            cond_dict = {
                "y": None,
                "c": cond_c,
                "cnn":cond_nn,
                "ct": None,
                "ctf": None,
                "cx": None,
                "cavg": cond_avg,
            }

            cond_sources = {
                "source_origin": u0, 
                "boundary_pointscloud": bnd_pointscloud,
                "x": x,
                "y": y,
            }

            return target, cond_dict, cond_sources

        else:
            raise ValueError(name)

    def latent_to_audio(self, x: Tensor) -> Tensor:
        r"""Ues vocoder to convert mel spectrogram to audio.

        Args:
            x: (b, 1, l1, l2)

        Outputs:
            y: (b, l)
        """
        l1 = 240
        x = rearrange(x, 'b c l1 l2 -> b (c l1 l2)', l1=l1)  
        
        return x