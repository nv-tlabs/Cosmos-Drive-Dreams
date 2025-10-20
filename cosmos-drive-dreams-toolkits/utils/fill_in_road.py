import torch
from math import ceil
from skimage import measure
from skspatial.objects import Plane

def voxel_downsample_torch_fast(points: torch.Tensor, voxel_size: list) -> torch.Tensor:
    """
    Fast voxel downsampling using PyTorch CUDA operations.
    For points that fall into the same voxel, randomly select one point.
    
    Args:
        points: torch.Tensor of shape (N, 3) containing 3D points
        voxel_size: list of [x, y, z] voxel sizes
        
    Returns:
        torch.Tensor of shape (M, 3) containing downsampled points, where M <= N
    """
    device = points.device
    
    # Convert points to voxel indices
    voxel_indices = torch.floor(points / torch.tensor(voxel_size, device=device)).long()
    
    # Shift coordinates to handle negative values
    min_vals = voxel_indices.min(dim=0)[0]
    voxel_indices = voxel_indices - min_vals
    
    # Calculate unique key for each point
    hash_multipliers = torch.tensor([1, 100000, 10000000000], device=device)
    voxel_keys = (voxel_indices * hash_multipliers).sum(dim=1)
    
    # Find unique voxels and counts
    unique_voxels, inverse_indices, counts = torch.unique(
        voxel_keys, 
        return_inverse=True, 
        return_counts=True
    )
    
    # Generate random values for each point
    rand_values = torch.rand(len(points), device=device)
    
    # Create a tensor to store the maximum random value for each voxel
    max_rand_per_voxel = torch.zeros_like(unique_voxels, dtype=torch.float32)
    
    # Use scatter_reduce to find the maximum random value in each voxel
    max_rand_per_voxel.scatter_reduce_(
        0, inverse_indices, rand_values, reduce='amax', include_self=False
    )
    
    # Create a mask for points that have the maximum random value in their voxel
    selected_mask = rand_values == max_rand_per_voxel[inverse_indices]
    
    # Return the selected points
    return points[selected_mask]

def upsample_plane_points(vv: torch.Tensor, 
                         uu: torch.Tensor, 
                         plane_params: torch.Tensor,
                         block_x_start: torch.Tensor,
                         block_y_start: torch.Tensor,
                         coarse_voxel_sizes: list,
                         fine_voxel_sizes: list) -> torch.Tensor:
    """
    Upsample points on a plane by subdividing each coarse grid cell into finer cells.
    
    Args:
        vv: torch.Tensor, v coordinates of original points in coarse grid
        uu: torch.Tensor, u coordinates of original points in coarse grid
        plane_params: torch.Tensor, [a,b,c,d] where plane equation is ax + by + cz + d = 0
        block_x_start: torch.Tensor, x coordinate of block start
        block_y_start: torch.Tensor, y coordinate of block start
        coarse_voxel_sizes: list, [x_size, y_size, z_size] of coarse voxels
        fine_voxel_sizes: list, [x_size, y_size, z_size] of fine voxels
        
    Returns:
        torch.Tensor of shape [num_points * Nx * Ny, 3] containing upsampled 3D points
    """
    device = vv.device
    
    # Calculate upsampling factors for x and y dimensions
    Nx = max(1, int(coarse_voxel_sizes[0] / fine_voxel_sizes[0]))
    Ny = max(1, int(coarse_voxel_sizes[1] / fine_voxel_sizes[1]))
    
    # Create offset grid for upsampling (in relative coordinates [0,1])
    offset_u = torch.linspace(0, 1, Nx+1)[:-1] + 1/(2*Nx)  # Center points in subdivisions
    offset_v = torch.linspace(0, 1, Ny+1)[:-1] + 1/(2*Ny)
    offset_u, offset_v = torch.meshgrid(offset_u, offset_v, indexing='xy')
    offset_u = offset_u.reshape(-1).to(device) - 0.5
    offset_v = offset_v.reshape(-1).to(device) - 0.5
    
    # Expand original indices
    uu_expanded = uu.unsqueeze(1) + offset_u.unsqueeze(0)  # [num_points, Nx*Ny]
    vv_expanded = vv.unsqueeze(1) + offset_v.unsqueeze(0)  # [num_points, Nx*Ny]
    
    # Flatten
    uu_flat = uu_expanded.reshape(-1)
    vv_flat = vv_expanded.reshape(-1)
    
    
    # Convert to world coordinates:
    # First multiply by coarse voxel size to get to the right grid cell
    # Then multiply the offset by fine voxel size to get the precise position within the cell
    x = uu_flat * coarse_voxel_sizes[0] + block_x_start
    y = vv_flat * coarse_voxel_sizes[1] + block_y_start
    
    # Calculate z using plane equation: ax + by + cz + d = 0
    # Therefore z = -(ax + by + d) / c
    z = -(plane_params[0] * x + plane_params[1] * y + plane_params[3]) / plane_params[2]

    # Stack the coordinates
    road_surface_points = torch.stack([x, y, z], dim=1)
    
    return road_surface_points

def estimate_road_surface_in_grid(road_edge_full, lane_full,
                                  block_x_idx, block_y_idx,
                                  blocks_x_start, blocks_x_end,
                                  blocks_y_start, blocks_y_end,
                                  voxel_sizes, 
                                  fine_voxel_sizes):
    """
    Args:
        road_edge: torch.Tensor
            shape (N_1, 3), road_edge points in world coordinates (confirmed)

        lane: torch.Tensor
            shape (N_2, 3), lane points in world coordinates (confirmed)

        voxel_sizes: list
            shape (3, ), voxel sizes in meters

             ---------------------
            |                     |  
            |                     |
            |                     |
            |      grid origin    |
            |          o --- x    |
            |          |          |
            |          y          |
            |                     |
            |                     |
             ---------------------
    """
    # Convert inputs to torch tensors if they aren't already
    device = torch.device("cuda")
    road_edge_full = torch.as_tensor(road_edge_full, device=device)
    lane_full = torch.as_tensor(lane_full, device=device)
    blocks_x_start = torch.as_tensor(blocks_x_start, device=device)
    blocks_x_end = torch.as_tensor(blocks_x_end, device=device)
    blocks_y_start = torch.as_tensor(blocks_y_start, device=device)
    blocks_y_end = torch.as_tensor(blocks_y_end, device=device)
    
    # retrieve exact block boundary coordinates
    block_x_start = blocks_x_start[block_x_idx]
    block_x_end = blocks_x_end[block_x_idx]
    block_y_start = blocks_y_start[block_y_idx]
    block_y_end = blocks_y_end[block_y_idx]

    # get block size
    block_size_x = block_x_end - block_x_start
    block_size_y = block_y_end - block_y_start

    # filter road edge that is out of the grid
    mask = (road_edge_full[:, 0] >= block_x_start) & (road_edge_full[:, 0] <= block_x_end) & \
           (road_edge_full[:, 1] >= block_y_start) & (road_edge_full[:, 1] <= block_y_end)
    road_edge = road_edge_full[mask]

    # filter lane that is out of the grid
    mask = (lane_full[:, 0] >= block_x_start) & (lane_full[:, 0] <= block_x_end) & \
           (lane_full[:, 1] >= block_y_start) & (lane_full[:, 1] <= block_y_end)
    lane = lane_full[mask]

    # expand to 3x3 block
    mask = (road_edge_full[:, 0] >= block_x_start - voxel_sizes[0]) & \
           (road_edge_full[:, 0] <= block_x_end + voxel_sizes[0]) & \
           (road_edge_full[:, 1] >= block_y_start - voxel_sizes[1]) & \
           (road_edge_full[:, 1] <= block_y_end + voxel_sizes[1])
    road_edge_full = road_edge_full[mask]

    mask = (lane_full[:, 0] >= block_x_start - voxel_sizes[0]) & \
           (lane_full[:, 0] <= block_x_end + voxel_sizes[0]) & \
           (lane_full[:, 1] >= block_y_start - voxel_sizes[1]) & \
           (lane_full[:, 1] <= block_y_end + voxel_sizes[1])
    lane_full = lane_full[mask]

    # if points are too few, we will not estimate the road surface for that block
    if road_edge.shape[0] < 3 or lane.shape[0] < 3:
        print(f"block {block_x_idx}_{block_y_idx} has too few points")
        return torch.zeros((0, 3), device=device)

    # create a BEV grid 
    bev_w = round(block_size_x.item() / voxel_sizes[0])
    bev_h = round(block_size_y.item() / voxel_sizes[1])

    bev_rasterize_map = torch.zeros((bev_h, bev_w), dtype=torch.uint8, device=device)

    # road edge points is dense, we can use it to rasterize the road edge
    road_edge_u = ((road_edge[:, 0] - block_x_start) // voxel_sizes[0]).long()
    road_edge_v = ((road_edge[:, 1] - block_y_start) // voxel_sizes[1]).long()
    
    # Add bounds checking for road edge points
    valid_road = (road_edge_u >= 0) & (road_edge_u < bev_w) & (road_edge_v >= 0) & (road_edge_v < bev_h)
    road_edge_u = road_edge_u[valid_road]
    road_edge_v = road_edge_v[valid_road]
    
    road_edge_uv = torch.stack([road_edge_u, road_edge_v], dim=1)
    road_edge_uv = torch.unique(road_edge_uv, dim=0)
    road_edge_u, road_edge_v = road_edge_uv[:, 0], road_edge_uv[:, 1]

    # also for lane points
    lane_u = ((lane[:, 0] - block_x_start) // voxel_sizes[0]).long()
    lane_v = ((lane[:, 1] - block_y_start) // voxel_sizes[1]).long()
    
    # Add bounds checking for lane points
    valid_lane = (lane_u >= 0) & (lane_u < bev_w) & (lane_v >= 0) & (lane_v < bev_h)
    lane_u = lane_u[valid_lane]
    lane_v = lane_v[valid_lane]
    
    # Check if we have enough valid points after bounds checking
    if len(lane_u) < 3 or len(road_edge_u) < 3:
        print(f"block {block_x_idx}_{block_y_idx} has too few valid points after bounds checking")
        return torch.zeros((0, 3), device=device)
    
    lane_uv = torch.stack([lane_u, lane_v], dim=1)
    lane_uv = torch.unique(lane_uv, dim=0)
    lane_u, lane_v = lane_uv[:, 0], lane_uv[:, 1]

    # draw road edge
    bev_rasterize_map[road_edge_v, road_edge_u] = 255

    # For connected components, we need to use CPU as skimage doesn't support GPU
    cc_image = torch.from_numpy(
        measure.label(bev_rasterize_map.cpu().numpy(), 
                     background=255, 
                     connectivity=1)
    ).to(device)
    
    # find the connected component that contains the lanes
    lane_cc = cc_image[lane_v, lane_u]
    lane_cc_unique = torch.unique(lane_cc)
    lane_cc_mask = torch.isin(cc_image, lane_cc_unique)

    # if lane_cc_mask is too large, handle subdivisions
    if lane_cc_mask.float().mean() > 0.7:
        # print(f"block {block_x_idx}_{block_y_idx} has too large lane_cc_mask")
        subdivide_num = 4
        sub_bev_w_idx = torch.linspace(0, bev_w, subdivide_num + 1, device=device).long()
        sub_bev_h_idx = torch.linspace(0, bev_h, subdivide_num + 1, device=device).long()

        for j in range(subdivide_num):
            for i in range(subdivide_num):
                lane_uv_in_sub_block = (lane_u >= sub_bev_w_idx[i]) & \
                                     (lane_u < sub_bev_w_idx[i+1]) & \
                                     (lane_v >= sub_bev_h_idx[j]) & \
                                     (lane_v < sub_bev_h_idx[j+1])
                                     
                road_edge_uv_in_sub_block = (road_edge_u >= sub_bev_w_idx[i]) & \
                                          (road_edge_u < sub_bev_w_idx[i+1]) & \
                                          (road_edge_v >= sub_bev_h_idx[j]) & \
                                          (road_edge_v < sub_bev_h_idx[j+1])
                
                sub_block_mask = lane_cc_mask[sub_bev_h_idx[j]:sub_bev_h_idx[j+1], 
                                            sub_bev_w_idx[i]:sub_bev_w_idx[i+1]]
                
                if (lane_uv_in_sub_block.sum() > 0 or road_edge_uv_in_sub_block.sum() > 0) and \
                    sub_block_mask.any():
                    
                    # Create meshgrid for distance calculations
                    sub_block_v = torch.arange(sub_bev_h_idx[j], sub_bev_h_idx[j+1], device=device)
                    sub_block_u = torch.arange(sub_bev_w_idx[i], sub_bev_w_idx[i+1], device=device)
                    
                    sub_block_x = sub_block_u * voxel_sizes[0] + block_x_start
                    sub_block_y = sub_block_v * voxel_sizes[1] + block_y_start
                    
                    grid_x, grid_y = torch.meshgrid(sub_block_x, sub_block_y, indexing='xy')
                    sub_block_xy = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)

                    # Calculate distances using GPU
                    lane_xy = lane_full[:, :2].unsqueeze(1)  # [N, 1, 2]
                    road_edge_xy = road_edge_full[:, :2].unsqueeze(1)  # [M, 1, 2]
                    
                    # Compute distances efficiently using broadcasting
                    lane_dist = torch.norm(sub_block_xy.unsqueeze(0) - lane_xy, dim=-1).min(dim=0)[0]
                    road_edge_dist = torch.norm(sub_block_xy.unsqueeze(0) - road_edge_xy, dim=-1).min(dim=0)[0]

                    # Create mask and reshape
                    mask = (lane_dist < road_edge_dist).view(
                        sub_bev_h_idx[j+1] - sub_bev_h_idx[j],
                        sub_bev_w_idx[i+1] - sub_bev_w_idx[i]
                    )
                    
                    # Update the lane_cc_mask
                    lane_cc_mask[sub_bev_h_idx[j]:sub_bev_h_idx[j+1], 
                               sub_bev_w_idx[i]:sub_bev_w_idx[i+1]] &= mask
                else:
                    lane_cc_mask[sub_bev_h_idx[j]:sub_bev_h_idx[j+1], 
                               sub_bev_w_idx[i]:sub_bev_w_idx[i+1]] = False

    # Sample points for plane fitting
    n_road_edge = min(1500, road_edge.shape[0])
    n_lane = min(1500, lane.shape[0])
    
    road_edge_idx = torch.randperm(road_edge.shape[0], device=device)[:n_road_edge]
    lane_idx = torch.randperm(lane.shape[0], device=device)[:n_lane]
    
    random_road_edge_pts = road_edge[road_edge_idx]
    random_lane_pts = lane[lane_idx]
    
    # For plane fitting, we need to use CPU as skspatial doesn't support GPU
    random_sample = torch.cat([random_road_edge_pts, random_lane_pts], dim=0).cpu().numpy()
    plane = Plane.best_fit(random_sample)
    a, b, c, d = plane.cartesian()  # ax + by + cz + d = 0
    
    # Move plane parameters to GPU
    plane_params = torch.tensor([a, b, c, d], device=device)

    # Get points where lane_cc_mask is True
    vv, uu = torch.where(lane_cc_mask)
    
    
    # Upsample the points using the plane equation
    road_surface_points = upsample_plane_points(
        vv=vv,
        uu=uu,
        plane_params=plane_params,
        block_x_start=block_x_start,
        block_y_start=block_y_start,
        coarse_voxel_sizes=voxel_sizes,
        fine_voxel_sizes=fine_voxel_sizes
    )

    return road_surface_points

def estimate_road_surface_in_world(road_edge, 
                                 lane,
                                 block_size = [40, 40],
                                 voxel_sizes = [0.4, 0.4, 0.2],
                                 fine_voxel_sizes = [0.02, 0.02, 0.2]):
    """
    Instead of ego car trajectory, we can use lane to estimate the road surface in world coordinates.
    """
    device = torch.device("cuda")
    road_edge = torch.as_tensor(road_edge, device=device)
    lane = torch.as_tensor(lane, device=device)
    
    # Calculate map boundaries
    map_x_min = lane[:, 0].min()
    map_x_max = lane[:, 0].max()
    map_y_min = lane[:, 1].min()
    map_y_max = lane[:, 1].max()

    # Calculate number of blocks
    block_x_num = ceil((map_x_max.item() - map_x_min.item()) / block_size[0])
    block_y_num = ceil((map_y_max.item() - map_y_min.item()) / block_size[1])

    print(f"block_x_num: {block_x_num}, block_y_num: {block_y_num}")

    # Create block boundaries
    blocks_x_start = torch.arange(block_x_num, device=device) * block_size[0] + map_x_min
    blocks_y_start = torch.arange(block_y_num, device=device) * block_size[1] + map_y_min
    blocks_x_end = blocks_x_start + block_size[0]
    blocks_y_end = blocks_y_start + block_size[1]

    # Create valid mask
    valid_mask = torch.zeros((block_y_num, block_x_num), dtype=torch.bool, device=device)
    
    for j in range(block_y_num):
        for i in range(block_x_num):
            mask_lane = (lane[:, 0] >= blocks_x_start[i]) & \
                       (lane[:, 0] < blocks_x_end[i]) & \
                       (lane[:, 1] >= blocks_y_start[j]) & \
                       (lane[:, 1] < blocks_y_end[j])
                       
            mask_road_edge = (road_edge[:, 0] >= blocks_x_start[i]) & \
                            (road_edge[:, 0] < blocks_x_end[i]) & \
                            (road_edge[:, 1] >= blocks_y_start[j]) & \
                            (road_edge[:, 1] < blocks_y_end[j])
                            
            if mask_lane.sum() > 0 and mask_road_edge.sum() > 0:
                valid_mask[j, i] = True

    print(f"valid blocks: {valid_mask.sum().item()}")

    # Process each block
    road_surface_points = []
    for j in range(block_y_num):
        for i in range(block_x_num):
            if not valid_mask[j, i]:
                continue

            points = estimate_road_surface_in_grid(
                road_edge, lane,
                i, j,
                blocks_x_start, blocks_x_end,
                blocks_y_start, blocks_y_end,
                voxel_sizes,
                fine_voxel_sizes
            )
            if points.shape[0] > 0:
                road_surface_points.append(points)

    if road_surface_points:
        road_surface_points = torch.cat(road_surface_points, dim=0)
    else:
        road_surface_points = torch.zeros((0, 3), device=device)

    return road_surface_points