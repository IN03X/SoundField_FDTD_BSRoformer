import torch
import torch.nn as nn
from torch import Tensor

from audio_flow.encoders.bigvgan import Mel_BigVGAN_44kHz


class Boundary2Field(nn.Module):
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

        if name in ["FDTD_2D"]:
 
            u0 = data["u0"].to(device) # (b, 1, t, f)
            bnd = data["bnd"].to(device) # (b, 1, t, f)
            t = data["t"].to(device) # (b, 1,)
            target = data["ut"].to(device)  # (b, 1, t, f)

            cond_c = t # (b, 1,)
            cond_tf = torch.cat([u0, bnd], dim=1) # (b, 2, t, f)

            cond_dict = {
                "y": None,
                "c": cond_c,
                "ct": None,
                "ctf": cond_tf,
                "cx": None
            }

            cond_sources = {
                "source_origin": u0, 
                "boundary": bnd,
                "t": t,
            }

            return target, cond_dict, cond_sources

        else:
            raise ValueError(name)

    def latent_to_audio(self, x: Tensor) -> Tensor:
        r"""Ues vocoder to convert mel spectrogram to audio.

        Args:
            x: (b, c, t, f)

        Outputs:
            y: (b, c, t, f)
        """
        return x