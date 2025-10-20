# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
from typing import Any, Dict, Tuple, TypeVar, Union

import numpy as np
import torch
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit
from cosmos_predict1.utils.camera.base import CameraBase

CropParams = TypeVar("CropParams")
ScaleParams = TypeVar("ScaleParams")

def compute_max_distance_to_border(image_size_component: float, principal_point_component: float) -> float:
    """Given an image size component (x or y) and corresponding principal point component (x or y),
    returns the maximum distance (in image domain units) from the principal point to either image boundary."""

    center = 0.5 * image_size_component
    if principal_point_component > center:
        return principal_point_component
    else:
        return image_size_component - principal_point_component


def compute_max_radius(image_size: np.ndarray, principal_point: np.ndarray) -> float:
    """Compute the maximum radius from the principal point to the image boundaries."""

    max_diag = np.array(
        [
            compute_max_distance_to_border(image_size[0], principal_point[0]),
            compute_max_distance_to_border(image_size[1], principal_point[1]),
        ]
    )
    return np.linalg.norm(max_diag).item()


def compute_ftheta_fw_mapping_max_angle(fwpoly: np.ndarray, max_radius_pixels: float):
    """
    Best guess of the valid domain of a forward mapping (ray to pixel).

    Args:
        fw_mapping (callable): Forward mapping function that maps angle to pixel radius.
        max_radius_pixels (float): Maximum radius in pixels.

    Returns:
        float: Maximum angle in radians.
    """

    # Constants
    MAX_ANGLE = np.pi  # Try up to 180 degrees (covers 360 degrees FOV if principal point is in the middle)
    DOMAIN_SAMPLE_COUNT = 2000  # Enough steps to get a fine enough resolution
    STEP_SIZE = MAX_ANGLE / DOMAIN_SAMPLE_COUNT

    angle = 0.0
    for i in range(1, DOMAIN_SAMPLE_COUNT + 1):
        next_angle = i * STEP_SIZE
        next_r = eval_polynomial(
            np.asarray(next_angle).reshape(1), fwpoly
        ).item()  # Forward polynomial maps angle to pixel radius

        # Compute the derivative at the next angle
        d_next_r = eval_polynomial_derivative(np.asarray(next_angle).reshape(1), fwpoly).item()

        if d_next_r <= 0.0:
            # Polynomial is monotonically increasing in valid domain, so derivative must be positive.
            # At this point, this is not the case -> stop
            break

        angle = next_angle
        if next_r > max_radius_pixels:
            # Now we're outside of the image pixels -> stop
            break

    return angle


def eval_polynomial(xs: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluates polynomial coeffs [D,] at given points [N,] using Horner scheme"""
    ret = np.zeros(len(xs), dtype=xs.dtype)

    for coeff in reversed(coeffs):
        ret = ret * xs + coeff

    return ret


def eval_polynomial_derivative(xs: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluates the derivative of polynomial coeffs [D,] at given points [N,] using Horner scheme"""
    ret = np.zeros(len(xs), dtype=xs.dtype)

    for i, coeff in enumerate(reversed(coeffs[:-1])):
        ret = ret * xs + coeff * (len(coeffs) - 1 - i)

    return ret


def approximate_polynomial_inverse(coeffs: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
    """
    Computes an approximate inverse polynomial in the provided range, fixing the first coefficient to zero.

    The fitting is performed via least squares on inverted sampled points (y, x) of the provided polynomial.

    Parameters:
    -----------
    coeffs : np.ndarray
        Coefficients of the polynomial to be inverted.
    range_min : float
        Minimum value of the range for sampling.
    range_max : float
        Maximum value of the range for sampling.

    Returns:
    --------
    np.ndarray
        Coefficients of the approximate inverse polynomial.

    Raises:
    -------
    ValueError
        If the polynomial degree is not supported (i.e., not 4 or 5).
    """

    def f4(x, b, x1, x2, x3, x4):
        """4th degree polynomial."""
        return b + x * (x1 + x * (x2 + x * (x3 + x * x4)))

    def f5(x, b, x1, x2, x3, x4, x5):
        """5th degree polynomial."""
        return b + x * (x1 + x * (x2 + x * (x3 + x * (x4 + x * x5))))

    SAMPLE_COUNT = 500
    samples_y = np.linspace(range_min, range_max, SAMPLE_COUNT)
    samples_x = eval_polynomial(samples_y, coeffs.astype(np.float64))  # use high-precision estimation

    # The constant term in the polynomial should be zero, so add the `bounds` condition
    if (poly_degree := len(coeffs) - 1) == 4:
        # Fit a 4th degree polynomial
        bounds = (
            [0, -np.inf, -np.inf, -np.inf, -np.inf],
            [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf],
        )
        inv_coeffs, _ = curve_fit(f4, samples_x, samples_y, bounds=bounds)
    elif poly_degree == 5:
        # Fit a 5th degree polynomial
        bounds = (
            [0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf, np.inf],
        )
        inv_coeffs, _ = curve_fit(f5, samples_x, samples_y, bounds=bounds)
    else:
        raise ValueError(f"Unsupported polynomial degree {poly_degree}")

    return np.array([np.float32(val) if i > 0 else 0 for i, val in enumerate(inv_coeffs)], dtype=np.float32)


class FThetaCamera(CameraBase):
    """Defines an FTheta camera model."""

    @classmethod
    def from_rig(cls, rig_file: str, sensor_name: str):
        """Helper method to initialize a new object using a rig file and the sensor's name.

        Args:
            rig_file (str): the rig file path.
            sensor_name (str): the name of the sensor. usually available sensors are:
                - camera:front:wide:120fov
                - camera:cross:left:120fov
                - camera:cross:right:120fov
                - camera:rear:left:70fov
                - camera:rear:right:70fov
                - camera:rear:tele:30fov
        Returns:
            FThetaCamera: the newly created object.
        """
        with open(rig_file, "r") as fp:
            rig = json.load(fp)

        # Parse the properties from the rig file
        sensors = rig["rig"]["sensors"]
        sensor = None
        sensor_found = False

        for sensor in sensors:
            if sensor["name"] == sensor_name:
                sensor_found = True
                break

        if not sensor_found:
            raise ValueError(f"The camera '{sensor_name}' was not found in the rig!")

        return cls.from_dict(sensor)

    @classmethod
    def from_dict(cls, rig_dict: Dict[str, Any]):
        """Helper method to initialize a new object using a dictionary of the rig.

        Args:
            rig_dict (dict): the sensor dictionary to initialize with.

        Returns:
            FThetaCamera: the newly created object.
        """
        cx, cy, width, height, bw_poly = FThetaCamera.get_ftheta_parameters_from_json(
            rig_dict
        )
        return cls(cx, cy, width, height, bw_poly)

    @classmethod
    def from_numpy(cls, intrinsics: np.ndarray, device=None):
        """Helper method to initialize a new object using an array of intrinsics.

        Args:
            intrinsics (np.ndarray): the intrinsics array. The ordering is expected to be
                "cx, cy, width, height, poly (more coefficients), is_bw_poly". This is the same ordering as the `intrinsics`
                property of this class.

        Returns:
            FThetaCaamera: the newly created object.
        """
        return cls(
            cx=intrinsics[0],
            cy=intrinsics[1],
            width=intrinsics[2],
            height=intrinsics[3],
            is_bw_poly=intrinsics[-1] > 0,
            poly=intrinsics[4:-1],
            device=device
        )

    def __init__(
        self, 
        cx: float, 
        cy: float, 
        width: int, 
        height: int, 
        poly: np.ndarray = None, 
        is_bw_poly: bool = True, 
        dtype=torch.float32, 
        device=None
    ):
        """The __init__ method.

        Args:
            cx (float): optical center x.
            cy (float): optical center y.
            width (int): the width of the image.
            height (int): the height of the image.
            is_bw_poly (bool): whether the poly is bw poly
            poly (np.ndarray): the polynomial of the FTheta model. Usually 5 coefficients.
        """
        self._rescale_width: int | None = None
        self._rescale_height: int | None = None
        self._center = np.asarray([cx, cy], dtype=np.float32)
        self._width = int(width)
        self._height = int(height)

        if is_bw_poly:
            self._bw_poly = Polynomial(poly)
            self._fw_poly = self._compute_fw_poly()
        else:
            self._fw_poly = Polynomial(poly)
            self._bw_poly = self._compute_bw_poly()

        # Other properties that need to be computed
        self._horizontal_fov = None
        self._vertical_fov = None
        self._max_angle = None
        self._max_ray_angle = None

        # Populate the array of intrinsics
        self._intrinsics = np.array([cx, cy, width, height, *poly, 1 if is_bw_poly else 0], dtype=np.float32)

        self._update_calibrated_camera()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.dtype = dtype

    @property
    def is_rescaled(self):
        return self._rescale_width is not None and self._rescale_height is not None

    def rescale(self, width: int, height: int):
        self._rescale_width = width
        self._rescale_height = height

    @staticmethod
    def get_ftheta_parameters_from_json(rig_dict: Dict[str, Any]) -> Tuple[Any]:
        """Helper method for obtaining FTheta camera model parameters from a rig dict.

        Args:
            rig_dict (Dict[str, Any]): the rig dictionary to parse.

        Raises:
            ValueError: if the provided rig is not supported.
            AssertionError: if the provided model is supported, but cannot be parsed properly.

        Returns:
            Tuple[Any]: the values `cx`, `cy`, `width`, `height` and `bw_poly` that were parsed.
        """
        props = rig_dict["properties"]

        if props["Model"] != "ftheta":
            raise ValueError("The given camera is not an FTheta camera")

        cx = float(props["cx"])
        cy = float(props["cy"])
        width = int(props["width"])
        height = int(props["height"])

        if "bw-poly" in props:  # Is this a regular rig?
            poly = props["bw-poly"]
        elif "polynomial" in props:  # Is this a VT rig?
            # VT rigs have a slightly different format, so need to handle these
            # specifically. Refer to the following thread for more details:
            # https://nvidia.slack.com/archives/C017LLEG763/p1633304770105300
            poly_type = props["polynomial-type"]
            assert poly_type == "pixeldistance-to-angle", (
                "Encountered an unsupported VT rig. Only `pixeldistance-to-angle` "
                f"polynomials are supported (got {poly_type}). Rig:\n{rig_dict}"
            )

            linear_c = float(props["linear-c"]) if "linear-c" in props else None
            linear_d = float(props["linear-d"]) if "linear-d" in props else None
            linear_e = float(props["linear-e"]) if "linear-e" in props else None

            # If we had all the terms present, sanity check to make sure they are [1, 0, 0]
            if linear_c is not None and linear_d is not None and linear_e is not None:
                assert (
                    linear_c == 1.0
                ), f"Expected `linear-c` term to be 1.0 (got {linear_c}. Rig:\n{rig_dict})"
                assert (
                    linear_d == 0.0
                ), f"Expected `linear-d` term to be 0.0 (got {linear_d}. Rig:\n{rig_dict})"
                assert (
                    linear_e == 0.0
                ), f"Expected `linear-e` term to be 0.0 (got {linear_e}. Rig:\n{rig_dict})"

            # If we're here, then it means we can parse the rig successfully.
            poly = props["polynomial"]
        else:
            raise ValueError(
                f"Unable to parse the rig. Only FTheta rigs are supported! Rig:\n{rig_dict}"
            )

        bw_poly = [np.float32(val) for val in poly.split()]
        return cx, cy, width, height, bw_poly

    @property
    def fov(self) -> tuple:
        """Returns a tuple of horizontal and vertical fov of the sensor."""
        if self._vertical_fov is None or self._horizontal_fov is None:
            self._compute_fov()
        return self._horizontal_fov, self._vertical_fov

    @property
    def width(self) -> int:
        """Returns the width of the sensor."""
        return self._width if not self.is_rescaled else self._rescale_width

    @property
    def height(self) -> int:
        """Returns the height of the sensor."""
        return self._height if not self.is_rescaled else self._rescale_height

    @property
    def center(self) -> np.ndarray:
        """Returns the center of the sensor."""
        assert not self.is_rescaled
        return self._center

    @property
    def intrinsics(self) -> np.ndarray:
        """Obtain an array of the intrinsics of this camera model.

        Returns:
            np.ndarray: an array of intrinsics. The ordering is "cx, cy, width, height, poly, is_bw_poly".
                dtype is np.float32.
        """
        assert not self.is_rescaled
        return self._intrinsics

    def __str__(self):
        """Returns a string representation of this object."""
        return (
            f"FTheta camera model:\n\t{self._bw_poly}\n\t"
            f"center={self._center}\n\twidth={self._width}\n\theight={self._height}\n\t"
            f"h_fov={np.degrees(self._horizontal_fov)}\n\tv_fov={np.degrees(self._vertical_fov)}"
        )

    def _update_calibrated_camera(self):
        """Updates the internals of this object after calulating various properties."""
        self._compute_fov()
        self._max_ray_angle = (self._max_angle).copy()
        is_fw_poly_slope_negative_in_domain = False
        ray_angle = (np.float32(self._max_ray_angle)).copy()
        deg2rad = np.pi / 180.0
        while ray_angle >= np.float32(0.0):
            temp_dval = self._fw_poly.deriv()(self._max_ray_angle).item()
            if temp_dval < 0:
                is_fw_poly_slope_negative_in_domain = True
            ray_angle -= deg2rad * np.float32(1.0)

        if is_fw_poly_slope_negative_in_domain:
            ray_angle = (np.float32(self._max_ray_angle)).copy()
            while ray_angle >= np.float32(0.0):
                ray_angle -= deg2rad * np.float32(1.0)
            raise ArithmeticError(
                "FThetaCamera: derivative of distortion within image interior is negative"
            )

        # Evaluate the forward polynomial at point (self._max_ray_angle, 0)
        # Also evaluate its derivative at the same point
        val = self._fw_poly(self._max_ray_angle).item()
        dval = self._fw_poly.deriv()(self._max_ray_angle).item()

        if dval < 0:
            raise ArithmeticError(
                "FThetaCamera: derivative of distortion at edge of image is negative"
            )

        self._max_ray_distortion = np.asarray([val, dval], dtype=np.float32)

    def _compute_fw_poly(self):
        """Computes the forward polynomial for this camera.

        This function is a replication of the logic in the following file from the DW repo:
        src/dw/calibration/cameramodel/CameraModels.cpp
        """
        max_value = compute_max_radius([self._width, self._height], self._center)
        coeffs = approximate_polynomial_inverse(self._bw_poly.coef, 0, max_value)

        # Return the polynomial and hardcode the bias value to 0
        return Polynomial(
            [np.float32(val) if i > 0 else 0 for i, val in enumerate(coeffs)]
        )

    
    def _compute_bw_poly(self):
        """Computes the backward polynomial for this camera.
        """
        max_pixel_radius = compute_max_radius([self._width, self._height], self._center)
        max_value = compute_ftheta_fw_mapping_max_angle(self._fw_poly.coef, max_pixel_radius)
        coeffs = approximate_polynomial_inverse(self._fw_poly.coef, 0, max_value)
        return Polynomial(
            [np.float32(val) if i > 0 else 0 for i, val in enumerate(coeffs)]
        )

    def _get_rays(self) -> torch.Tensor:
        """
        Returns:
            rays: (H, W, 3), normalized camera rays in opencv convention
            
          z (front)
         /    
        o ------> x (right)
        |
        v y (down)
        """
        u = torch.arange(self.width, dtype=torch.int32, device=self.device)
        v = torch.arange(self.height, dtype=torch.int32, device=self.device)
        u, v = torch.meshgrid(u, v, indexing='xy') # must pass indexing='xy'
        uv_coords = torch.stack([u, v], dim=-1).reshape(-1, 2) # shape (H, W, 2)
        
        # call pixel2ray to get the rays
        rays = self.pixel2ray(uv_coords.cpu().numpy())[0]
        rays = torch.tensor(rays, device=self.device, dtype=self.dtype).reshape(self.height, self.width, 3)
        return rays


    def pixel2ray(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Backproject 2D pixels into 3D rays.

        Args:
            x (np.ndarray): the pixels to backproject. Size of (n_points, 2), where the first
                column contains the `x` values, and the second column contains the `y` values.

        Returns:
            rays (np.ndarray): the backprojected 3D rays. Size of (n_points, 3).
            valid (np.ndarray): bool flag indicating the validity of each backprojected pixel.
        """
        # Make sure x is n x 2
        if np.ndim(x) == 1:
            x = x[np.newaxis, :]

        if self.is_rescaled:
            x = x * np.array(
                [self._width / self._rescale_width, self._height / self._rescale_height]
            ).astype(np.float32)

        # Fix the type
        x = x.astype(np.float32)
        xd = x - self._center
        xd_norm = np.linalg.norm(xd, axis=1, keepdims=True)
        alpha = self._bw_poly(xd_norm)
        sin_alpha = np.sin(alpha)

        rx = sin_alpha * xd[:, 0:1] / xd_norm
        ry = sin_alpha * xd[:, 1:] / xd_norm
        rz = np.cos(alpha)

        rays = np.hstack((rx, ry, rz))
        # special case: ray is perpendicular to image plane normal
        valid = (xd_norm > np.finfo(np.float32).eps).squeeze()
        rays[~valid, :] = (0, 0, 1)  # This is what DW sets these rays to

        # note:
        # if constant coefficient of bwPoly is non-zero,
        # the resulting ray might not be normalized.
        return rays, valid
    
    def ray2pixel(self, rays: np.ndarray) -> np.ndarray:
        """Project 3D rays to 2D pixel coordinates.

        Args:
            rays (np.ndarray): the rays. shape: (M, 3)

        Returns:
            result (np.ndarray): the projected pixel coordinates. shape: (M, 2)
        """
        # Make sure the input shape is (n_points, 3)
        if np.ndim(rays) == 1:
            rays = rays[np.newaxis, :]

        # Fix the type
        rays = rays.astype(np.float32)
        # TODO(restes) combine 2 and 3 column norm for rays?
        xy_norm = np.linalg.norm(rays[:, :2], axis=1, keepdims=True)
        cos_alpha = rays[:, 2:] / np.linalg.norm(rays, axis=1, keepdims=True)

        alpha = np.empty_like(cos_alpha)
        cos_alpha_condition = np.logical_and(
            cos_alpha > np.float32(-1.0), cos_alpha < np.float32(1.0)
        ).squeeze()
        alpha[cos_alpha_condition] = np.arccos(cos_alpha[cos_alpha_condition])
        alpha[~cos_alpha_condition] = xy_norm[~cos_alpha_condition]

        delta = np.empty_like(cos_alpha)
        alpha_cond = alpha <= self._max_ray_angle
        delta[alpha_cond] = self._fw_poly(alpha[alpha_cond])
        # For outside the model (which need to do linear extrapolation)
        delta[~alpha_cond] = (
            self._max_ray_distortion[0]
            + (alpha[~alpha_cond] - self._max_ray_angle) * self._max_ray_distortion[1]
        )

        # Determine the bad points with a norm of zero, and avoid division by zero
        bad_norm = xy_norm <= 0
        xy_norm[bad_norm] = 1
        delta[bad_norm] = 0
        # compute pixel relative to center
        scale = delta / xy_norm
        pixel = scale * rays

        # Handle the edge cases (ray along image plane normal)
        edge_case_cond = (xy_norm <= np.float32(0.0)).squeeze()
        pixel[edge_case_cond, :] = rays[edge_case_cond, :]
        result = pixel[:, :2] + self._center

        if self.is_rescaled:
            result = result * np.array(
                [self._rescale_width / self._width, self._rescale_height / self._height]
            ).astype(np.float32)

        return result
    
    def ray2pixel_np(self, rays: np.ndarray) -> np.ndarray:
        # call ray2pixel
        return self.ray2pixel(rays)
    
    def ray2pixel_torch(self, rays: torch.Tensor) -> torch.Tensor:
        device = rays.device
        dtype = rays.dtype
        return torch.tensor(self.ray2pixel(rays.cpu().numpy()), device=device, dtype=dtype)

    def _get_pixel_fov(self, pt: np.ndarray) -> float:
        """Gets the FOV for a given point. Used internally for FOV computation.

        Args:
            pt (np.ndarray): 2D pixel.

        Returns:
            fov (float): the FOV of the pixel.
        """
        ray, _ = self.pixel2ray(pt)
        fov = np.arctan2(np.linalg.norm(ray[:, :2], axis=1), ray[:, 2])
        return fov

    def _compute_fov(self):
        """Computes the FOV of this camera model."""
        max_x = self._width - 1
        max_y = self._height - 1

        point_left = np.asarray([0, self._center[1]], dtype=np.float32)
        point_right = np.asarray([max_x, self._center[1]], dtype=np.float32)
        point_top = np.asarray([self._center[0], 0], dtype=np.float32)
        point_bottom = np.asarray([self._center[0], max_y], dtype=np.float32)

        fov_left = self._get_pixel_fov(point_left)
        fov_right = self._get_pixel_fov(point_right)
        fov_top = self._get_pixel_fov(point_top)
        fov_bottom = self._get_pixel_fov(point_bottom)

        self._vertical_fov = fov_top + fov_bottom
        self._horizontal_fov = fov_left + fov_right
        self._compute_max_angle()

    def _compute_max_angle(self):
        """Computes the maximum ray angle for this camera."""
        max_x = self._width - 1
        max_y = self._height - 1

        p = np.asarray(
            [[0, 0], [max_x, 0], [0, max_y], [max_x, max_y]], dtype=np.float32
        )

        self._max_angle = max(
            max(self._get_pixel_fov(p[0, ...]), self._get_pixel_fov(p[1, ...])),
            max(self._get_pixel_fov(p[2, ...]), self._get_pixel_fov(p[3, ...])),
        )

    def is_ray_inside_fov(self, ray: np.ndarray) -> bool:
        """Determines whether a given ray is inside the FOV of this camera.

        Args:
            ray (np.ndarray): the 3D ray.

        Returns:
            bool: whether the ray is inside the FOV.
        """
        if np.ndim(ray) == 1:
            ray = ray[np.newaxis, :]

        ray_angle = np.arctan2(np.linalg.norm(ray[:, :2], axis=1), ray[:, 2])
        return ray_angle <= self._max_angle



def rays_to_pixels_batch(
    rays: torch.Tensor, cam_intrinsic: torch.Tensor
) -> torch.Tensor:
    """Project 3D rays to 2D pixel coordinates.

    Computes image projection for 3D points in an
    F-theta camera.
    More information about the projection is in the below link:
    https://confluence.nvidia.com/display/DS/Camera+intrinsic+models
    Args:
        rays (torch.Tensor): 3D rays tensor with size (B, H, W, 3, N).
            where N is the number of vertices.
        cam_intrinsic (torch.Tensor): camera intrinsic. tensor with size 9.
            In particular, the element 0 is img_cx, 1 is img_cy,
            2 is img_width, 3 is img_height, 4:9 is img_bw_poly.

    Returns:
        pixels (torch.Tensor): the projected pixel coordinates.
            Tensor is of size (B, H, W, 2, N).
    """
    cam_intrinsic = cam_intrinsic.cpu().detach().numpy()
    img_cx = cam_intrinsic[0]
    img_cy = cam_intrinsic[1]
    img_width = cam_intrinsic[2]
    img_height = cam_intrinsic[3]
    img_bw_poly = cam_intrinsic[4:]
    ftheta_model = FThetaCamera(img_cx, img_cy, img_width, img_height, img_bw_poly)
    # Computes the forward polynomial for this camera
    fw_poly = list(ftheta_model._compute_fw_poly())
    device = rays.device
    max_ray_angle = torch.tensor(ftheta_model._max_ray_angle, device=device)
    max_ray_distortion = torch.tensor(ftheta_model._max_ray_distortion, device=device)

    ray_x = rays[:, :, :, 0, :]
    ray_y = rays[:, :, :, 1, :]
    ray_z = rays[:, :, :, 2, :]
    ray_xy_norm = torch.linalg.norm(torch.stack([ray_x, ray_y], dim=3), dim=3)
    # Compute the angle of ray with the optical axis
    alpha = (math.pi / 2.0) - torch.atan2(ray_z, ray_xy_norm)

    delta_valid = (
        fw_poly[0]
        + fw_poly[1] * alpha
        + fw_poly[2] * alpha**2
        + fw_poly[3] * alpha**3
        + fw_poly[4] * alpha**4
    )
    # For outside the model (which need to do linear extrapolation)
    delta_invalid = (
        max_ray_distortion[0] + (alpha - max_ray_angle) * max_ray_distortion[1]
    )
    delta = torch.where(alpha <= max_ray_angle, delta_valid, delta_invalid)

    # Determine the bad points with a norm of zero, and avoid division by zero
    delta = torch.where(ray_xy_norm > 0.0, delta, torch.tensor(0.0, device=device))
    ray_xy_norm = torch.where(
        ray_xy_norm > 0.0, ray_xy_norm, torch.tensor(1.0, device=device)
    )

    # compute pixel relative to center
    scale = delta / ray_xy_norm

    pixel_x = img_cx + scale * ray_x
    pixel_y = img_cy + scale * ray_y
    pixels = torch.stack([pixel_x, pixel_y], dim=3)

    return pixels


def convert_to_2d_camera_model(
    xyz: np.ndarray, ft_camera: FThetaCamera
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a series of points in xyz to 2D camera coordinates.

    Args:
        xyz: Batch of points [x, 3] in shape.
        ft_camera: The camera model to use for the conversion to 2D coordinates.

    Returns:
        Points in camera uv coordinates.
    """
    P = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    flu = P @ xyz.T
    uv = ft_camera.ray2pixel(flu.T)
    return uv


def unit_test_downsample():
    cx = 1931.257881559029
    cy = 1509.477889795519
    height = 2160
    width = 3840

    # front wide 120's bw poly
    bw_poly = [0, 0.0005280342432281367, 3.440655735426777e-09, -5.025598287293601e-12, 5.143362581269205e-15, 0]
    ftheta_camera = FThetaCamera(cx, cy, width, height, bw_poly)
    print(ftheta_camera._fw_poly)

    # downsample 2x 
    bw_poly_downsample = [p * 2**i for i, p in enumerate(bw_poly)]
    ftheta_camera_downsample = FThetaCamera(cx / 2, cy / 2, width / 2, height / 2, bw_poly_downsample)
    print(ftheta_camera_downsample._fw_poly)
    

def unit_test_bw_poly_inverse():
    cx = 1931.257881559029
    cy = 1509.477889795519
    height = 2160
    width = 3840
    
    # front wide 120's bw poly
    bw_poly = [0, 0.0005280342432281367, 3.440655735426777e-09, -5.025598287293601e-12, 5.143362581269205e-15, 0]
    ftheta_camera = FThetaCamera(cx, cy, width, height, bw_poly)
    fw_poly = ftheta_camera._fw_poly.coef
    print(fw_poly)

    # use fw_poly to compute the inverse
    ftheta_camera_fw_input = FThetaCamera(cx, cy, width, height, fw_poly, is_bw_poly=False)
    bw_poly = ftheta_camera_fw_input._bw_poly.coef
    print(bw_poly)
    

if __name__ == "__main__":
    unit_test_downsample()