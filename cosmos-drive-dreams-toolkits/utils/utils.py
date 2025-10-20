# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all
# intellectual property and proprietary rights in and to this software,
# related documentation and any modifications thereto. Any use,
# reproduction, disclosure or distribution of this software and related
# documentation without an express license agreement from NVIDIA
# CORPORATION & AFFILIATES is strictly prohibited.

import torch, numpy as np, re
from einops import rearrange
from matplotlib import cm

def to_numpy_array(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")


def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def transform_points_torch(points: torch.Tensor, tfm: torch.Tensor) -> torch.Tensor:
    """
    Args:
        points: (M, 3)
        tfm: (4, 4)
    Returns:
        points_transformed: (M, 3)
    """
    points_transformed = tfm[:3, :3] @ points.T + tfm[:3, 3].reshape(-1, 1)
    return points_transformed.T


def apply_color_map_to_image(
    x: torch.Tensor,
    mask: torch.Tensor = None,
    color_map: str = "Spectral",
) -> torch.Tensor:
    if color_map == "gray":
        mapped = np.repeat(x.detach().clip(min=0, max=1).cpu().numpy()[..., np.newaxis], 3, axis=-1)
    else:
        cmap = cm.get_cmap(color_map)
        # Convert to NumPy so that Matplotlib color maps can be used.
        mapped = cmap(x.detach().float().clip(min=0, max=1).cpu().numpy())[..., :3]
    image = torch.tensor(mapped, device=x.device, dtype=x.dtype)
    if mask is not None:
        image[mask] = torch.tensor([0.82, 0.82, 0.82], device=x.device, dtype=x.dtype)  # Set masked areas to light gray

    return rearrange(image, "... h w c -> ... c h w")


def colorcode_depth_maps(result, near=None, far=None, cmap="turbo"):
    """
    Input: B x H x W
    Output: B x 3 x H x W, normalized to [0, 1]
    """
    mask = result == 0
    n_frames = result.shape[0]
    
    far = result[n_frames // 2].view(-1).quantile(0.99).log() if far is None else far    
    if near is None:
        try:
            near = result[n_frames // 2][result[n_frames // 2] > 0].quantile(0.01).log()
        except:
            print("No valid depth values found.")
            near = torch.zeros_like(far)
            
    result = 1 - (result.log() - near) / (far - near)
    return apply_color_map_to_image(result, mask, cmap)