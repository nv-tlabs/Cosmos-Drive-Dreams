import numpy as np
import tensorflow as tf
import click
import imageio as imageio_v1

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Union
from termcolor import cprint
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.utils import frame_utils
from google.protobuf import json_format
from utils.wds_utils import write_to_tar, encode_dict_to_npz_bytes
from utils.bbox_utils import interpolate_pose

WaymoProto2SemanticLabel = {
    label_pb2.Label.Type.TYPE_UNKNOWN: "Unknown",
    label_pb2.Label.Type.TYPE_VEHICLE: "Car",
    label_pb2.Label.Type.TYPE_PEDESTRIAN: "Pedestrian",
    label_pb2.Label.Type.TYPE_SIGN: "Sign",
    label_pb2.Label.Type.TYPE_CYCLIST: "Cyclist",
}

CameraNames = ['front', 'front_left', 'front_right', 'side_left', 'side_right']

SourceFps = 10 # waymo's recording fps
TargetFps = 30 # cosmos's expected fps
IndexScaleRatio = int(TargetFps / SourceFps)

if int(tf.__version__.split(".")[0]) < 2:
    tf.enable_eager_execution()

# make sure the GPU memory is not exhausted
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print(e)

def get_camera_name(name_int) -> str:
    return dataset_pb2.CameraName.Name.Name(name_int)


def get_lidar_name(name_int) -> str:
    return dataset_pb2.LaserName.Name.Name(name_int)


def convert_waymo_intrinsics(output_root: Path, clip_id: str, dataset: tf.data.TFRecordDataset):
    """
    read the first frame and convert the intrinsics to wds format

    Minimal required format:
        sample['pinhole_intrinsic.{camera_name}.npy'] = np.ndarray with shape (4, 4)
    """
    sample = {'__key__': clip_id}

    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        for camera_calib in frame.context.camera_calibrations:
            camera_name = get_camera_name(camera_calib.name).lower()

            intrinsic = camera_calib.intrinsic
            fx, fy, cx, cy = intrinsic[:4]
            w, h = camera_calib.width, camera_calib.height

            sample[f'pinhole_intrinsic.{camera_name}.npy'] = \
                np.array([fx, fy, cx, cy, w, h])

        write_to_tar(sample, output_root / 'pinhole_intrinsic' / f'{clip_id}.tar')

        # only process the first frame
        break


def convert_waymo_hdmap(output_root: Path, clip_id: str, dataset: tf.data.TFRecordDataset):
    """
    read the first frame and convert the hdmap to wds format. One map type corresponds to one webdataset file.

    Minimal required format:
        sample['{hdmap_name}.json'] = {
            'labels': [
                {
                    'labelData': {
                        'shape3d': {
                            'polyline3d' / 'surface' : {'vertices': list of list of float}
                        }
                    }
                },
                ...
            ]
        }

    hdmap_name should be consistent with cosmos's name convention.
    """
    def hump_to_underline(hump_str):
        import re
        return re.sub(r'([a-z])([A-Z])', r'\1_\2', hump_str).lower()

    hdmap_names_polyline = ["lane", "road_line", "road_edge"]
    hdmap_names_polygon = ["crosswalk", "speed_bump", "driveway"]
    
    hdmap_name_to_data = {}
    for hdmap_name in hdmap_names_polyline + hdmap_names_polygon:
        hdmap_name_to_data[hump_to_underline(hdmap_name)] = []

    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        map_features_list = json_format.MessageToDict(frame)['mapFeatures']

        for hdmap_content in map_features_list:
            hdmap_name = list(hdmap_content.keys())
            hdmap_name.remove("id")
            hdmap_name = hdmap_name[0]
            hdmap_name_lower = hump_to_underline(hdmap_name)

            hdmap_data = hdmap_content[hdmap_name]
            if hdmap_name_lower in hdmap_names_polyline:
                hdmap_data = hdmap_data['polyline']
                polyline = [[point['x'], point['y'], point['z']] for point in hdmap_data]
                hdmap_name_to_data[hdmap_name_lower].append(polyline)
            elif hdmap_name_lower in hdmap_names_polygon:
                hdmap_data = hdmap_data['polygon']
                polygon = [[point['x'], point['y'], point['z']] for point in hdmap_data]
                hdmap_name_to_data[hdmap_name_lower].append(polygon)
            else:
                print(f"Unkown hdmap item name: {hdmap_name}，skip this item")

        # only process the first frame
        break

    # convert to cosmos's name convention for easier processing
    hdmap_name_to_cosmos = {
        'lane': 'lanes',
        'road_line': 'lanelines',
        'road_edge': 'road_boundaries',
        'crosswalk': 'crosswalks',
        'speed_bump': None,
        'driveway': None
    }

    for hdmap_name, hdmap_data in hdmap_name_to_data.items():
        hdmap_name_in_cosmos = hdmap_name_to_cosmos[hdmap_name]
        if hdmap_name_in_cosmos is None:
            continue

        if hdmap_name in hdmap_names_polyline:
            vertex_indicator = 'polyline3d'
        else:
            vertex_indicator = 'surface'

        # to match cosmos format, the easiest way is to add 'vertices' key for the polyline or polygon
        sample = {'__key__': clip_id, f'{hdmap_name_in_cosmos}.json': {'labels': []}}

        for each_polyline_or_polygon in hdmap_data:
            sample[f'{hdmap_name_in_cosmos}.json']['labels'].append({
                'labelData': {
                    'shape3d': {
                        vertex_indicator: {
                            'vertices': each_polyline_or_polygon
                        }
                    }
                }
            })

        write_to_tar(sample, output_root / f'3d_{hdmap_name_in_cosmos}' / f'{clip_id}.tar')


def convert_waymo_pose(output_root: Path, clip_id: str, dataset: tf.data.TFRecordDataset):
    """
    read all frames and convert the pose to wds format. interpolate the pose to the target fps

    Minimal required format:
        sample_camera_to_world['{frame_idx:06d}.pose.{camera_name}.npy'] = np.ndarray with shape (4, 4). opencv convention
        sample_vehicle_to_world['{frame_idx:06d}.vehicle_pose.npy'] = np.ndarray with shape (4, 4). flu convention
    """
    sample_camera_to_world = {'__key__': clip_id}
    sample_vehicle_to_world = {'__key__': clip_id}

    camera_name_to_camera_to_vehicle = {}

    # get camera_to_vehicle from the first frame
    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        for camera_calib in frame.context.camera_calibrations:
            camera_name = get_camera_name(camera_calib.name).lower()
            camera_to_vehicle = np.array(camera_calib.extrinsic.transform).reshape((4, 4)) # FLU convention
            camera_name_to_camera_to_vehicle[camera_name] = camera_to_vehicle

        # only process the first frame
        break

    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        for image_data in frame.images:
            camera_name = get_camera_name(image_data.name).lower()
            vehicle_to_world = np.array(image_data.pose.transform).reshape((4, 4))
            camera_to_vehicle = camera_name_to_camera_to_vehicle[camera_name]
            camera_to_world = vehicle_to_world @ camera_to_vehicle # FLU convention
            camera_to_world_opencv = np.concatenate(
                [-camera_to_world[:, 1:2], -camera_to_world[:, 2:3], camera_to_world[:, 0:1], camera_to_world[:, 3:4]],
                axis=1
            )
            sample_camera_to_world[f"{frame_idx * IndexScaleRatio:06d}.pose.{camera_name}.npy"] = camera_to_world_opencv

        sample_vehicle_to_world[f"{frame_idx * IndexScaleRatio:06d}.vehicle_pose.npy"] = vehicle_to_world

    # interpolate the pose to the target fps
    # source index: 0,    1,    2,    3, ..., 10
    # target index: 0,1,2,3,4,5,6,7,8,9, ..., 30,31,32
    max_target_frame_idx = frame_idx * IndexScaleRatio

    # interpolate the vehicle pose to the target fps
    for target_frame_idx in range(max_target_frame_idx):
        if f"{target_frame_idx:06d}.vehicle_pose.npy" not in sample_vehicle_to_world:
            nearest_prev_frame_idx = target_frame_idx // IndexScaleRatio * IndexScaleRatio
            nearest_prev_frame_pose = sample_vehicle_to_world[f"{nearest_prev_frame_idx:06d}.vehicle_pose.npy"]
            nearest_next_frame_idx = (target_frame_idx // IndexScaleRatio + 1) * IndexScaleRatio
            nearest_next_frame_pose = sample_vehicle_to_world[f"{nearest_next_frame_idx:06d}.vehicle_pose.npy"]
            sample_vehicle_to_world[f"{target_frame_idx:06d}.vehicle_pose.npy"] = \
                interpolate_pose(nearest_prev_frame_pose, nearest_next_frame_pose, (target_frame_idx - nearest_prev_frame_idx) / IndexScaleRatio)

    # add the last two frames
    approx_motion = sample_vehicle_to_world[f"{max_target_frame_idx:06d}.vehicle_pose.npy"] - sample_vehicle_to_world[f"{max_target_frame_idx - 1:06d}.vehicle_pose.npy"]
    approx_motion[:3, :3] = 0
    sample_vehicle_to_world[f"{max_target_frame_idx + 1:06d}.vehicle_pose.npy"] = sample_vehicle_to_world[f"{max_target_frame_idx:06d}.vehicle_pose.npy"] + approx_motion
    sample_vehicle_to_world[f"{max_target_frame_idx + 2:06d}.vehicle_pose.npy"] = sample_vehicle_to_world[f"{max_target_frame_idx:06d}.vehicle_pose.npy"] + 2 * approx_motion

    # interpolate the camera pose to the target fps
    for camera_name in CameraNames:
        for target_frame_idx in range(max_target_frame_idx):
            if f"{target_frame_idx:06d}.pose.{camera_name}.npy" not in sample_camera_to_world:
                nearest_prev_frame_idx = target_frame_idx // IndexScaleRatio * IndexScaleRatio
                nearest_prev_frame_pose = sample_camera_to_world[f"{nearest_prev_frame_idx:06d}.pose.{camera_name}.npy"]
                nearest_next_frame_idx = (target_frame_idx // IndexScaleRatio + 1) * IndexScaleRatio
                nearest_next_frame_pose = sample_camera_to_world[f"{nearest_next_frame_idx:06d}.pose.{camera_name}.npy"]
                sample_camera_to_world[f"{target_frame_idx:06d}.pose.{camera_name}.npy"] = \
                    interpolate_pose(nearest_prev_frame_pose, nearest_next_frame_pose, (target_frame_idx - nearest_prev_frame_idx) / IndexScaleRatio)

        # add the last two frames
        approx_motion = sample_camera_to_world[f"{max_target_frame_idx:06d}.pose.{camera_name}.npy"] - sample_camera_to_world[f"{max_target_frame_idx - 1:06d}.pose.{camera_name}.npy"]
        approx_motion[:3, :3]  = 0
        sample_camera_to_world[f"{max_target_frame_idx + 1:06d}.pose.{camera_name}.npy"] = sample_camera_to_world[f"{max_target_frame_idx:06d}.pose.{camera_name}.npy"] + approx_motion
        sample_camera_to_world[f"{max_target_frame_idx + 2:06d}.pose.{camera_name}.npy"] = sample_camera_to_world[f"{max_target_frame_idx:06d}.pose.{camera_name}.npy"] + 2 * approx_motion

    write_to_tar(sample_camera_to_world, output_root / 'pose' / f'{clip_id}.tar')
    write_to_tar(sample_vehicle_to_world, output_root / 'vehicle_pose' / f'{clip_id}.tar')


def convert_waymo_timestamp(output_root: Path, clip_id: str, dataset: tf.data.TFRecordDataset):
    """
    read all frames and convert the timestamp to wds format.
    """
    sample = {'__key__': clip_id}
    
    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        timestamp_micros = frame.timestamp_micros
        sample[f"{frame_idx * IndexScaleRatio:06d}.timestamp_micros.txt"] = str(timestamp_micros)
        
    write_to_tar(sample, output_root / 'timestamp' / f'{clip_id}.tar')


def convert_waymo_bbox(output_root: Path, clip_id: str, dataset: tf.data.TFRecordDataset):
    """
    read all frames and convert the bbox to wds format

    Minimal required format:
        sample['{frame_idx:06d}.all_object_info.json'] = {
            'object_id 1' : {
                'object_to_world' : np.ndarray with shape (4, 4),
                'object_lwh' : np.ndarray with shape (3,),
                'object_is_moving' : bool,
                'object_type' : str
            },
            'object_id 2' : {
                ...
            },
            ...
        }
    """
    sample = {'__key__': clip_id}
    min_moving_speed = 0.2

    valid_bbox_types = [
        label_pb2.Label.Type.TYPE_VEHICLE,
        label_pb2.Label.Type.TYPE_PEDESTRIAN,
        label_pb2.Label.Type.TYPE_CYCLIST
    ]

    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        vehicle_to_world = np.array(frame.pose.transform).reshape((4, 4))
        sample[f"{frame_idx * IndexScaleRatio:06d}.all_object_info.json"] = {}

        for label in frame.laser_labels:
            if label.type not in valid_bbox_types:
                continue

            if not label.camera_synced_box.ByteSize():
                continue

            object_id = label.id
            object_type = WaymoProto2SemanticLabel[label.type]

            center_in_vehicle = np.array([label.camera_synced_box.center_x, label.camera_synced_box.center_y, label.camera_synced_box.center_z, 1]).reshape((4, 1))
            center_in_world = vehicle_to_world @ center_in_vehicle
            heading = label.camera_synced_box.heading
            rotation_in_vehicle = R.from_euler("xyz", [0, 0, heading], degrees=False).as_matrix()
            rotation_in_world = vehicle_to_world[:3, :3] @ rotation_in_vehicle

            object_to_world = np.eye(4)
            object_to_world[:3, :3] = rotation_in_world
            object_to_world[:3, 3] = center_in_world.flatten()[:3]

            object_lwh = np.array([label.camera_synced_box.length, label.camera_synced_box.width, label.camera_synced_box.height])
            
            speed = np.sqrt(label.metadata.speed_x**2 + label.metadata.speed_y**2 + label.metadata.speed_z**2)
            object_is_moving = bool(speed > min_moving_speed)

            sample[f"{frame_idx * IndexScaleRatio:06d}.all_object_info.json"][object_id] = {
                'object_to_world': object_to_world.tolist(),
                'object_lwh': object_lwh.tolist(),
                'object_is_moving': object_is_moving,
                'object_type': object_type
            }

    write_to_tar(sample, output_root / 'all_object_info' / f'{clip_id}.tar')


def convert_waymo_lidar(output_root: Path, clip_id: str, dataset: tf.data.TFRecordDataset):
    """
    read all frames and convert the lidar to wds format.

    Minimal required format:
        sample['{frame_idx:06d}.lidar.npz'] = {
            'xyz' : np.ndarray with shape (N, 3),
            'lidar_to_world' : np.ndarray with shape (4, 4)
        }
    """
    sample = {'__key__': clip_id}
    cprint('reading lidar and converting to RDS-HQ format, this may take a few minutes...', 'yellow')

    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        vehicle_to_world = np.array(frame.pose.transform).reshape((4, 4))
        
        range_images, camera_projections,  _, range_image_top_pose = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        points, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose
        )
        points_ri2, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=1
        )

        # 3d points in vehicle frame. 
        points = [np.concatenate([p1, p2]) for p1, p2 in zip(points, points_ri2)] 

        lidar_ids = [calib.name for calib in frame.context.laser_calibrations]
        lidar_ids.sort()

        for lidar_id, lidar_points in zip(lidar_ids, points):
            lidar_name = get_lidar_name(lidar_id)

            # we only want the TOP lidar
            if lidar_name == 'TOP':
                sample[f"{frame_idx * IndexScaleRatio:06d}.lidar_raw.npz"] = encode_dict_to_npz_bytes(
                    {
                        'xyz': lidar_points,
                        'lidar_to_world': vehicle_to_world
                    }
                )

    write_to_tar(sample, output_root / 'lidar_raw' / f'{clip_id}.tar')


def convert_waymo_image(output_root: Path, clip_id: str, dataset: tf.data.TFRecordDataset, single_camera: bool = False  ):
    """
    read all frames and convert the images to video format.
    """
    cprint('reading image and converting to video, this may take a while...', 'yellow')

    camera_name_to_image_sequence = {}
    for camera_name in CameraNames:
        camera_name_to_image_sequence[camera_name] = []

    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        for image_data in frame.images:
            camera_name = get_camera_name(image_data.name).lower()
            image_data_bytes = image_data.image
            image_data_tf_tensor = tf.image.decode_jpeg(image_data_bytes)
            image_data_numpy = image_data_tf_tensor.numpy()
            camera_name_to_image_sequence[camera_name].append(image_data_numpy)
    
    for camera_name, image_sequence in camera_name_to_image_sequence.items():
        # waymo is recorded at 10 Hz
        if single_camera and camera_name != 'front':
           continue
        output_video_path = (output_root / f"pinhole_{camera_name}" / f'{clip_id}.mp4')
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        writer = imageio_v1.get_writer(
            output_video_path.as_posix(),
            fps=SourceFps,
            macro_block_size=None,  # This makes sure num_frames is correct (by default it is rounded to 16x).
        )
        for image in image_sequence:
            writer.append_data(image)
        writer.close()

def convert_waymo_tfrecord_to_wds(
    waymo_tfrecord_filename: Union[str, Path],
    output_wds_path: Union[str, Path],
    single_camera: bool = False
):
    waymo_tfrecord_path = Path(waymo_tfrecord_filename)
    clip_id = waymo_tfrecord_path.stem.lstrip('segment-').rstrip('_with_camera_labels')
    output_wds_path = Path(output_wds_path)

    if not waymo_tfrecord_path.exists():
        raise FileNotFoundError(f"Waymo tfrecord file not found: {waymo_tfrecord_path}")
    
    if (output_wds_path / 'lidar_raw' / f"{clip_id}.tar").exists():
        print(f"Skipping {clip_id} because it already exists")
        return

    dataset = tf.data.TFRecordDataset(waymo_tfrecord_path, compression_type="")

    convert_waymo_hdmap(output_wds_path, clip_id, dataset)
    convert_waymo_intrinsics(output_wds_path, clip_id, dataset)
    convert_waymo_pose(output_wds_path, clip_id, dataset)
    convert_waymo_timestamp(output_wds_path, clip_id, dataset)
    convert_waymo_bbox(output_wds_path, clip_id, dataset)
    convert_waymo_image(output_wds_path, clip_id, dataset, single_camera)
    convert_waymo_lidar(output_wds_path, clip_id, dataset)

@click.command()
@click.option("--waymo_tfrecord_root", "-i", type=str, help="Waymo tfrecord root")
@click.option("--output_wds_path", "-o", type=str, help="Output wds path")
@click.option("--num_workers", "-n", type=int, default=1, help="Number of workers")
@click.option("--single_camera", "-s", type=bool, default=False, help="Convert only front camera")
def main(waymo_tfrecord_root: str, output_wds_path: str, num_workers: int, single_camera: bool):
    all_filenames = list(Path(waymo_tfrecord_root).glob("*.tfrecord"))
    print(f"Found {len(all_filenames)} tfrecords")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                convert_waymo_tfrecord_to_wds,
                waymo_tfrecord_filename=filename,
                output_wds_path=output_wds_path,
                single_camera=single_camera
            ) 
            for filename in all_filenames
        ]
        
        for future in tqdm(
            as_completed(futures), 
            total=len(all_filenames),
            desc="Converting tfrecords"
        ):
            try:
                future.result() 
            except Exception as e:
                print(f"Failed to convert due to error: {e}")

if __name__ == "__main__":
    main()
