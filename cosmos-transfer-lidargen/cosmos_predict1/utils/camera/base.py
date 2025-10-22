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


import torch
import numpy as np
import cv2
from cosmos_predict1.utils.misc import make_sure_numpy, make_sure_torch
from typing import List, Tuple, Union
from abc import ABC, abstractmethod

DEPTH_MAX = 122.5

def interpolate_polyline_to_points(polyline, voxel_size=0.025):
    """
    polyline:
        numpy.ndarray, shape (N, 3) or list of points 

    Returns:
        points: numpy array, shape (interpolate_num*N, 3)
    """
    def interpolate_points(previous_vertex, vertex):
        """
        Args:
            previous_vertex: (x, y, z)
            vertex: (x, y, z)

        Returns:
            points: numpy array, shape (interpolate_num, 3)
        """
        interpolate_num = int(np.linalg.norm(np.array(vertex) - np.array(previous_vertex)) / voxel_size)
        interpolate_num = max(interpolate_num, 2)
        
        # interpolate between previous_vertex and vertex
        x = np.linspace(previous_vertex[0], vertex[0], num=interpolate_num)
        y = np.linspace(previous_vertex[1], vertex[1], num=interpolate_num)
        z = np.linspace(previous_vertex[2], vertex[2], num=interpolate_num)
        return np.stack([x, y, z], axis=1)

    points = []
    previous_vertex = None
    for idx, vertex in enumerate(polyline):
        if idx == 0:
            previous_vertex = vertex
            continue
        else:
            points.extend(interpolate_points(previous_vertex, vertex))
            previous_vertex = vertex

    return np.array(points)


class CameraBase:
    def __init__(self):
        pass

    @abstractmethod
    def ray2pixel_torch(self, rays: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def ray2pixel_np(self, rays: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def _get_rays(self) -> torch.Tensor:
        raise NotImplementedError
    
    def transform_points_torch(self, points: torch.Tensor, tfm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (M, 3)
            tfm: (4, 4)
        Returns:
            points_transformed: (M, 3)
        """
        transformed_points = tfm[:3, :3] @ points.T + tfm[:3, 3].unsqueeze(-1)
        return transformed_points.T
    
    def transform_points_np(self, points: np.ndarray, tfm: np.ndarray) -> np.ndarray:
        """
        Args:
            points: (M, 3)
            tfm: (4, 4)
        Returns:
            points_transformed: (M, 3)
        """
        transformed_points = tfm[:3, :3] @ points.T + tfm[:3, 3].reshape(-1, 1)
        return transformed_points.T
    
    def _get_rays_posed(self, camera_poses: torch.Tensor):
        """
        Args:
            camera_poses: (N, 4, 4)
        Returns:
            ray_o: (N, H, W, 3), camera origin
            ray_d: (N, H, W, 3), camera rays
        """
        rays_in_cam = self._get_rays() # shape (H, W, 3)
        rays_d_in_world = torch.einsum('bij,hwj->bhwi', camera_poses[:, :3, :3], rays_in_cam) # shape (N, H, W, 3)
        rays_o_in_world = camera_poses[:, :3, 3].unsqueeze(-2).unsqueeze(-2).expand_as(rays_d_in_world) # shape (N, H, W, 3)
        
        return rays_o_in_world, rays_d_in_world

    def _clip_polyline_to_image_plane(self, points_in_cam: np.ndarray) -> np.ndarray:
        """
        Args:
            points_in_cam: np.ndarray
                shape: (M, 3), a polyline, they are connected.
        Returns:
            points: np.ndarray
                shape: (M', 3), a polyline, but we clip the points to positive z if the points are behind the camera.
        """        
        depth = points_in_cam[:, 2]
        # go through all the edges of the polyline. 
        eps = 1e-1
        cam_coords_cliped = []
        for i in range(len(points_in_cam) - 1):
            pt1 = points_in_cam[i]
            pt2 = points_in_cam[i+1]

            if depth[i] >= 0 and depth[i+1] >= 0:
                cam_coords_cliped.append(pt1)
            elif depth[i] < 0 and depth[i+1] < 0:
                continue
            else:
                # clip the line to the image boundary
                if depth[i] >= 0:
                    # add the first point
                    cam_coords_cliped.append(pt1)

                    # calculate the intersection point and add it
                    t = (- pt2[2]) / (pt1[2] - pt2[2]) + eps
                    inter_pt = pt2 + t * (pt1 - pt2)
                    cam_coords_cliped.append(inter_pt)
                else:
                    # calculate the intersection point and add it
                    t = (- pt1[2]) / (pt2[2] - pt1[2]) + eps
                    inter_pt = pt1 + t * (pt2 - pt1)
                    cam_coords_cliped.append(inter_pt)

        # handle the last point, if its depth > 0 and not already added
        if depth[-1] >= 0:
            cam_coords_cliped.append(points_in_cam[-1])

        cam_coords_cliped = np.stack(cam_coords_cliped, axis=0) # shape (M', 3)
        
        return cam_coords_cliped

    """
    Projection related functions
    """
    def distance_to_zbuffer(self, distance_map: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Args:
            distance_map: (N, H, W) or (H, W), distance to the camera
        Returns:
            zbuffer_map: (N, H, W) or (H, W), 0 means no depth value
        """
        rays = self._get_rays() # normalized camera rays, shape (H, W, 3)

        return distance_map * rays[..., 2].expand_as(distance_map)


    def zbuffer_to_distance(self, zbuffer_map: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Args:
            zbuffer_map: (N, H, W) or (H, W), depth value
        Returns:
            distance_map: (N, H, W) or (H, W), distance to the camera
        """
        rays = self._get_rays()

        return zbuffer_map / rays[..., 2].expand_as(zbuffer_map)
    
    def get_zbuffer_map_from_points(self, camera_poses: Union[torch.Tensor, np.ndarray], points: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(camera_poses, np.ndarray):
            return self.get_zbuffer_map_from_points_np(camera_poses, points)
        else:
            return self.get_zbuffer_map_from_points_torch(camera_poses, points)

    def get_zbuffer_map_from_points_torch(self, camera_poses: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_pose: (N, 4, 4) or (4, 4)
            points: (M, 3), in world coordinate
        Returns:
            zbuffer_map: (N, H, W) or (H, W), 0 means no depth value
        """
        depth_images = []
        camera_poses = make_sure_torch(camera_poses).to(self.device).to(self.dtype)
        points = make_sure_torch(points).to(self.device).to(self.dtype)

        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses.unsqueeze(0)

        for camera_to_world in camera_poses:
            points_in_cam = self.transform_points_torch(points, torch.inverse(camera_to_world))
            uv_coords = self.ray2pixel_torch(points_in_cam)
            depth = points_in_cam[:, 2]
            valid_depth_mask = depth > 0

            u_round = torch.round(uv_coords[:, 0]).long()
            v_round = torch.round(uv_coords[:, 1]).long()

            valid_uv_mask = (u_round >= 0) & (u_round < self.width) & (v_round >= 0) & (v_round < self.height)
            valid_mask = valid_depth_mask & valid_uv_mask

            u_valid = u_round[valid_mask]
            v_valid = v_round[valid_mask]
            z_valid = depth[valid_mask]

            indices = v_valid * self.width + u_valid

            depth_image = torch.full((self.height, self.width), float('inf')).to(depth).flatten()
            depth_image = depth_image.scatter_reduce_(0, indices, z_valid, "amin")
            depth_image = depth_image.view(self.height, self.width)
            depth_mask = torch.isfinite(depth_image)

            # change inf to 0
            depth_image[~depth_mask] = 0

            depth_images.append(depth_image)

        depth_images = torch.stack(depth_images, dim=0)

        if depth_images.shape[0] == 1:
            depth_images = depth_images.squeeze(0)

        return depth_images


    def get_zbuffer_map_from_points_np(self, camera_poses: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Args:
            camera_pose: (N, 4, 4) or (4, 4)
            points: (M, 3), in world coordinate
        Returns:
            zbuffer_map: (N, H, W) or (H, W), 0 means no depth value
        """
        depth_images = []
        camera_poses = make_sure_numpy(camera_poses)
        points = make_sure_numpy(points)

        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses[np.newaxis, ...]

        for camera_to_world in camera_poses:
            points_in_cam = self.transform_points_np(points, np.linalg.inv(camera_to_world))
            uv_coords = self.ray2pixel_np(points_in_cam)
            depth = points_in_cam[:, 2]
            valid_depth_mask = depth > 0

            u_round = np.round(uv_coords[:, 0]).astype(np.int32)
            v_round = np.round(uv_coords[:, 1]).astype(np.int32)

            valid_uv_mask = (u_round >= 0) & (u_round < self.width) & (v_round >= 0) & (v_round < self.height)
            valid_mask = valid_depth_mask & valid_uv_mask

            u_valid = u_round[valid_mask]
            v_valid = v_round[valid_mask]
            z_valid = depth[valid_mask]

            depth_image = np.zeros((self.height, self.width))
            # use np.minimum.at to update the depth image
            np.minimum.at(depth_image, (v_valid, u_valid), z_valid)

            depth_images.append(depth_image)

        depth_images = np.stack(depth_images, axis=0)

        if depth_images.shape[0] == 1:
            depth_images = depth_images[0]

        return depth_images


    def get_distance_map_from_points(self, camera_poses: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_pose: (N, 4, 4) or (4, 4)
            points: (M, 3), in world coordinate
        Returns:
            distance_map: (N, H, W) or (H, W), 0 means no depth value
        """
        depth_map = self.get_zbuffer_map_from_points(camera_poses, points) # shape (N, H, W) or (H, W)
        return self.zbuffer_to_distance(depth_map)


    def get_distance_map_from_voxel(self, camera_poses: torch.Tensor, voxel_grid) -> torch.Tensor:
        """
        Args:
            camera_pose: (N, 4, 4) or (4, 4)
            voxel_grid: fvdb.GridBatch

        Returns:
            distance_map: (N, H, W) or (H, W), 0 means no depth value
        """
        camera_poses = make_sure_torch(camera_poses).to(self.device).to(self.dtype)
        voxel_grid = voxel_grid.to(self.device).to(self.dtype)

        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses.unsqueeze(0)

        rays_o, rays_d = self._get_rays_posed(camera_poses)
        N, H, W = rays_o.shape[:3]

        segment = voxel_grid.segments_along_rays(rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), 1, eps=1e-1) 
        pixel_hit = segment.joffsets[1:] - segment.joffsets[:-1]
        pixel_hit = pixel_hit.view(N, H, W).float()
        distance = segment.jdata[:, 0] # [N_hit,]

        distance_map = torch.zeros((N, H, W)).to(distance.device)
        distance_map[pixel_hit > 0] = distance

        if N == 1:
            distance_map = distance_map.squeeze(0)

        return distance_map

    def get_zbuffer_map_from_voxel(self, camera_poses: torch.Tensor, voxel_grid) -> torch.Tensor:
        """
        Args:
            camera_pose: (N, 4, 4) or (4, 4)
            voxel_grid: fvdb.GridBatch

        Returns:
            zbuffer_map: (N, H, W) or (H, W), 0 means no depth value
        """
        distance_map = self.get_distance_map_from_voxel(camera_poses, voxel_grid)
        return self.distance_to_zbuffer(distance_map)


    def get_semantic_map_from_voxel(self, camera_poses: torch.Tensor, voxel_grid, voxel_semantic: torch.Tensor, background_semantic: int = 0) -> torch.Tensor:
        """
        Args:
            camera_pose: (N, 4, 4) or (4, 4)
            voxel_grid: fvdb.GridBatch
            voxel_semantic: torch.Tensor, (#voxel, )
        
        Returns:
            semantic_map: (N, H, W) or (H, W)
        """
        camera_poses = make_sure_torch(camera_poses).to(self.device).to(self.dtype)
        voxel_grid = voxel_grid.to(self.device)
        voxel_semantic = voxel_semantic.to(self.device)

        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses.unsqueeze(0)

        rays_o, rays_d = self._get_rays_posed(camera_poses)
        N, H, W = rays_o.shape[:3]

        vox, times = voxel_grid.voxels_along_rays(rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), 1, eps=1e-1, return_ijk=False) 

        pixel_hit = times.joffsets[1:] - times.joffsets[:-1]
        pixel_hit = pixel_hit.view(N, H, W).bool() # (N, H, W) 0,1 mask

        # semantic 
        hit_voxel_semantic = voxel_semantic[vox.jdata] # [N_hit, C]

        # fill the semantic to the image plane. 0 refers to UNDEFINED, which is the sky / background.
        semantic_rasterize = torch.full((N, H, W), background_semantic).to(hit_voxel_semantic)
        semantic_rasterize[pixel_hit] = hit_voxel_semantic

        if N == 1:
            semantic_rasterize = semantic_rasterize.squeeze(0)  # (H, W)

        return semantic_rasterize

    """
    Drawing related functions
    """
    def draw_points(
            self, 
            camera_poses: Union[torch.Tensor, np.ndarray], 
            points: Union[torch.Tensor, np.ndarray], 
            colors: Union[torch.Tensor, np.ndarray, None] = None, 
            radius: int = 1,
            fast_impl_when_radius_gt_1: bool = True
        ) -> np.ndarray:
        """
        Args:
            camera_poses: torch.Tensor or np.ndarray
                shape: (N, 4, 4) or (4, 4) 
            points: torch.Tensor or np.ndarray
                shape: (M, 3) 
            colors: torch.Tensor or np.ndarray or None
                shape: (M, 3) in uint8
            radius: int, 
                radius of the point
            fast_impl_when_radius_gt_1: bool, 
                if True, use cv2.circle to draw the point when radius > 1
        Returns:
            canvas: np.ndarray
                shape: (N, H, W, 3) or (H, W, 3)
                dtype: np.uint8
        """
        draw_images = []
        camera_poses = make_sure_numpy(camera_poses)
        points = make_sure_numpy(points)

        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses[np.newaxis, ...]

        if colors is not None:
            colors = make_sure_numpy(colors)
        else:
            colors = np.tile([[255, 0, 0]], (points.shape[0], 1))

        for camera_to_world in camera_poses:
            points_in_cam = self.transform_points_np(points, np.linalg.inv(camera_to_world))
            uv_coords = self.ray2pixel_np(points_in_cam)
            depth = points_in_cam[:, 2]
            valid_depth_mask = depth > 0

            u_round = np.round(uv_coords[:, 0]).astype(np.int32)
            v_round = np.round(uv_coords[:, 1]).astype(np.int32)

            valid_uv_mask = (u_round >= 0) & (u_round < self.width) & (v_round >= 0) & (v_round < self.height)
            valid_mask = valid_depth_mask & valid_uv_mask

            u_valid = u_round[valid_mask]
            v_valid = v_round[valid_mask]
            z_valid = depth[valid_mask]
            colors_valid = colors[valid_mask]

            sorted_indices = np.argsort(z_valid, axis=0)[::-1]
            u_valid = u_valid[sorted_indices]
            v_valid = v_valid[sorted_indices]
            colors_valid = colors_valid[sorted_indices]

            if radius > 1 and fast_impl_when_radius_gt_1 is False:
                canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                for u, v, color in zip(u_valid, v_valid, colors_valid):
                    cv2.circle(canvas, (u.item(), v.item()), radius, color.tolist(), -1)
                canvas = np.array(canvas, dtype=np.uint8)

            # radius = 1 or we want fast impl
            else: 
                canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                canvas[v_valid, u_valid] = colors_valid # fill from the farthest point to the nearest point

                # use fast impl when radius > 1
                if radius > 1:
                    canvas_accum = np.zeros_like(canvas)
                    i_shifts = np.arange(-radius//2, radius//2+1)
                    j_shifts = np.arange(-radius//2, radius//2+1)
                    for i in i_shifts:
                        for j in j_shifts:
                            # use torch.roll to shift the canvas 
                            canvas_shifted = np.roll(canvas, shift=(i, j), axis=(0, 1))
                            canvas_accum = np.maximum(canvas_accum, canvas_shifted)
                    canvas = canvas_accum

            draw_images.append(canvas)

        draw_images = np.stack(draw_images, axis=0)

        if draw_images.shape[0] == 1:
            draw_images = draw_images[0]

        return draw_images

    def draw_line_depth(
            self, 
            camera_poses: Union[torch.Tensor, np.ndarray], 
            polylines: List, 
            radius: int = 8,
            colors: np.ndarray = None,
            segment_interval: float = 0,
        ) -> np.ndarray:
        """
        draw lines on the image, and the drawed pixel value is related to the depth of the points.
        The polyline can be out of boundary, use cv2.clipLine to clip the line to the image boundary, or abandon the line.
        Then use cv2.line to draw the line.

        Args:
            camera_poses: torch.Tensor or np.ndarray
                shape: (N, 4, 4) or (4, 4)
            polylines: list of list of points, 
                each point is in 3D (x, y, z)
            radius: int, 
                radius of the drawn circle
            colors: np.ndarray, 
                shape: (3, ), dtype: np.uint8
            segment_interval: float, 
                if > 0, the polyline is segmented into segments with the interval

        Returns:
            draw_images: np.ndarray
                shape: (N, H, W, 3) or (H, W, 3)
                dtype: np.uint8
        """
        draw_images = []
        camera_poses = make_sure_numpy(camera_poses)

        if colors is None:
            colors = np.array([255, 255, 255])

        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses[np.newaxis, ...]
        
        for camera_to_world in camera_poses:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            H, W = canvas.shape[:2]

            for polyline in polylines:
                if len(polyline) < 2:
                    continue
                if isinstance(polyline, list):
                    polyline = np.array(polyline)

                if segment_interval > 0:
                    polyline = interpolate_polyline_to_points(polyline, segment_interval)
                
                points_in_cam = self.transform_points_np(polyline, np.linalg.inv(camera_to_world))
                uv_coords = self.ray2pixel_np(points_in_cam)
                depth = points_in_cam[:, 2]

                u_round = np.round(uv_coords[:, 0]).astype(np.int32)
                v_round = np.round(uv_coords[:, 1]).astype(np.int32)
                valid_uv_mask = (u_round >= 0) & (u_round < W) & (v_round >= 0) & (v_round < H)

                # filter out the polyline if all points are out of the image boundary
                if (~valid_uv_mask).all():
                    continue

                # if depth all greater than DEPTH_MAX, skip
                if depth.min() > DEPTH_MAX:
                    continue

                for i in range(len(u_round) - 1):
                    if depth[i] < 0 and depth[i+1] < 0:
                        continue
                    
                    if depth[i] * depth[i+1] < 0:
                        # if the two points are on different sides of the camera, we first clip the 3d point in the back to the camera plane + epsilon
                        # and then reproject it to the image plane, calculate the uv coordinate
                        pt1 = points_in_cam[i]
                        pt2 = points_in_cam[i+1]

                        # make sure pt1 is in front of the camera, pt2 is behind the camera
                        if depth[i] < 0:
                            pt1, pt2 = pt2, pt1

                        # clip the line to the image boundary
                        eps = 2e-1
                        t = (- pt2[2]) / (pt1[2] - pt2[2]) + eps
                        pt2 = t * pt1 + (1 - t) * pt2

                        # project the point to the image plane
                        pt1_norm = pt1[:3] / pt1[2]
                        pt2_norm = pt2[:3] / pt2[2]

                        pixel1 = self.ray2pixel_np(pt1_norm)[0] 
                        pixel2 = self.ray2pixel_np(pt2_norm)[0]
                    else:
                        pixel1 = np.array([u_round[i], v_round[i]])
                        pixel2 = np.array([u_round[i+1], v_round[i+1]])

                    try:
                        clipped, pixel1, pixel2 = \
                            cv2.clipLine((0, 0, W, H), pixel1.astype(np.int32), pixel2.astype(np.int32))
                    except:
                        breakpoint()

                    depth_mean = (depth[i] + depth[i+1]) / 2
                    depth_mean = np.clip(depth_mean, 0, DEPTH_MAX)
                    fill_weight = (2 * (DEPTH_MAX - depth_mean)) / 255
                    fill_value = (fill_weight * colors).astype(np.uint8).tolist()

                    cv2.line(canvas, tuple(pixel1), tuple(pixel2), fill_value, radius)

            draw_images.append(canvas)

        draw_images = np.stack(draw_images, axis=0)

        if draw_images.shape[0] == 1 and len(draw_images.shape) == 3:
            draw_images = draw_images[0]
        
        return draw_images


    def draw_hull_depth(
            self, 
            camera_poses: Union[torch.Tensor, np.ndarray], 
            hulls: List,
            colors: np.ndarray = None
        ) -> torch.Tensor:
        """
        draw hulls on the image, and the drawed pixel value is related to the depth of the points.
        The hull can be out of boundary, use cv2.clipLine to clip the line to the image boundary, or abandon the line.
        Then use cv2.line to draw the line.

        Args:
            camera_poses: torch.Tensor or np.ndarray
                shape: (N, 4, 4) or (4, 4)
            hulls: list of list of points, 
                each point is in 3D (x, y, z)
            colors: np.ndarray, 
                shape: (3, ), dtype: np.uint8

        Returns:
            draw_images: (N, H, W, 3) or (H, W, 3), image with hulls drawn
        """
        draw_images = []
        camera_poses = make_sure_numpy(camera_poses)

        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses[np.newaxis, ...]
        
        for camera_to_world in camera_poses:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            H, W = canvas.shape[:2]

            for hull in hulls:
                if len(hull) < 3:
                    continue
                
                points_in_cam = self.transform_points_np(hull, np.linalg.inv(camera_to_world))
                uv_coords = self.ray2pixel_np(points_in_cam).astype(np.int32)
                depth = points_in_cam[:, 2]
                valid_depth_mask = depth > 0

                u_round = uv_coords[:, 0]
                v_round = uv_coords[:, 1]
                valid_uv_mask = (u_round >= 0) & (u_round < W) & (v_round >= 0) & (v_round < H)
                valid_mask = valid_depth_mask & valid_uv_mask

                # filter out the polyline if all points are out of the image boundary
                if not valid_mask.any():
                    continue

                # if depth all greater than DEPTH_MAX, skip
                if depth.min() > DEPTH_MAX:
                    continue

                # project again with clipped points
                points_in_cam_clipped = self._clip_polyline_to_image_plane(points_in_cam)
                uv_coords = self.ray2pixel_np(points_in_cam_clipped).astype(np.int32)
                depth_mean = points_in_cam_clipped[:, 2].mean()

                # cv2.clipLine to limit the points to the image boundary
                uv_coords_line_clipped = []
                for i in range(len(uv_coords) - 1):
                    pixel1 = uv_coords[i]
                    pixel2 = uv_coords[i+1]
                    clipped, pixel1, pixel2 = cv2.clipLine((0, 0, W, H), pixel1, pixel2)
                    uv_coords_line_clipped.append(pixel1)
                    uv_coords_line_clipped.append(pixel2)

                hull_in_uv = np.array(uv_coords_line_clipped).astype(np.int32)

                # create convex hull and draw on the image
                convex_hull_in_uv = cv2.convexHull(hull_in_uv.astype(np.int32)) # shape (N, 1, 2)
                fill_weight = (2 * (DEPTH_MAX - depth_mean)) / 255
                fill_value = (fill_weight * colors).astype(np.uint8).tolist()
                cv2.fillPoly(canvas, [convex_hull_in_uv], fill_value)

            draw_images.append(canvas)

        draw_images = np.stack(draw_images, axis=0)

        if draw_images.shape[0] == 1 and len(draw_images.shape) == 3:
            draw_images = draw_images[0]

        return draw_images