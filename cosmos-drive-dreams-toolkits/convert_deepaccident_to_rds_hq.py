import cv2
import pickle
import click
import numpy as np
import imageio as imageio_v1

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from functools import partial

from utils.wds_utils import write_to_tar
from utils.wds_utils import encode_dict_to_npz_bytes
from utils.bbox_utils import interpolate_pose

OBJECT_MAP = {
    "car": "Car",
    "truck": "Car",
    "bus": "Car",
    "pedestrian": "Pedestrian",
    "motorcycle": "Cyclist",
    "bicycle": "Cyclist",
}

CameraNameMap = {
    "Camera_Front": "front",
    "Camera_FrontLeft": "front_left",
    "Camera_FrontRight": "front_right",
    "Camera_Back": "back",
    "Camera_BackLeft": "back_left",
    "Camera_BackRight": "back_right",
}

SourceFps = 10
TargetFps = 30
IndexScaleRatio = TargetFps // SourceFps
TimestampInterval = int(1e6 / SourceFps)

# convert_intrinsics()
def convert_intrinsics(output_root: Path, clip_id: str, calib_folder: Path, image_root: Path):
    """
    Convert DeepAccident intrinsics into RDS-HQ format.

    Output:
        [fx, fy, cx, cy, width, height]
    """
    sample = {"__key__": clip_id}
    calib_file = sorted(calib_folder.glob("*.pkl"))[0]
    with open(calib_file, "rb") as f:
        calib = pickle.load(f)
    #
    # obtain image resolution
    #
    first_camera = list(CameraNameMap.keys())[0]
    image_dir = image_root / first_camera / clip_id
    first_image = sorted(image_dir.glob("*.jpg"))[0]
    img = cv2.imread(str(first_image))
    height, width = img.shape[:2]
    #
    # convert every camera
    #
    for deep_name, rds_name in CameraNameMap.items():
        K = calib[f"intrinsic_{deep_name}"]
        cx = float(K[0,0])
        fx = float(K[0,1])
        cy = float(K[1,0])
        fy = float(-K[1,2])
        sample[f"pinhole_intrinsic.{rds_name}.npy"] = np.array([fx, fy, cx, cy, width, height], dtype=np.float32)

    write_to_tar(sample, output_root / "pinhole_intrinsic" / f"{clip_id}.tar",)

# convert_pose()
def convert_pose(
    output_root: Path,
    clip_id: str,
    calib_folder: Path,
):
    """
    Convert DeepAccident poses into RDS-HQ format.

    Outputs
    -------
    pose/
        camera_to_world (OpenCV convention)

    vehicle_pose/
        ego(vehicle)_to_world
    """
    sample_camera = {"__key__": clip_id}
    sample_vehicle = {"__key__": clip_id}
    calib_files = sorted(calib_folder.glob("*.pkl"))

    for frame_idx, calib_file in enumerate(calib_files):
        with open(calib_file, "rb") as f:
            calib = pickle.load(f)

        ego_to_world = calib["ego_to_world"].astype(np.float32)
        sample_vehicle[
            f"{frame_idx * IndexScaleRatio:06d}.vehicle_pose.npy"
        ] = ego_to_world

        ego_to_lidar = np.linalg.inv(
            calib["lidar_to_ego"]
        ).astype(np.float32)

        for deep_name, rds_name in CameraNameMap.items():

            camera_to_lidar = np.linalg.inv(
                calib[f"lidar_to_{deep_name}"]
            ).astype(np.float32)

            #
            # Camera -> World
            #
            camera_to_world = (ego_to_world @ ego_to_lidar @ camera_to_lidar)
            camera_to_world_opencv = np.concatenate(
                [
                    -camera_to_world[:, 1:2],
                    -camera_to_world[:, 2:3],
                     camera_to_world[:, 0:1],
                     camera_to_world[:, 3:4],
                ],
                axis=1,
            ).astype(np.float32)

            sample_camera[
                f"{frame_idx * IndexScaleRatio:06d}.pose.{rds_name}.npy"
            ] = camera_to_world_opencv

    max_target = (len(calib_files) - 1) * IndexScaleRatio

    #
    # vehicle pose
    #
    for target_idx in range(max_target):
        key = f"{target_idx:06d}.vehicle_pose.npy"
        if key in sample_vehicle:
            continue

        prev_idx = (target_idx // IndexScaleRatio) * IndexScaleRatio
        next_idx = prev_idx + IndexScaleRatio

        sample_vehicle[key] = interpolate_pose(
            sample_vehicle[f"{prev_idx:06d}.vehicle_pose.npy"],
            sample_vehicle[f"{next_idx:06d}.vehicle_pose.npy"],
            (target_idx - prev_idx) / IndexScaleRatio,
        ).astype(np.float32)

    #
    # camera pose
    #
    for camera in CameraNameMap.values():
        for target_idx in range(max_target):
            key = f"{target_idx:06d}.pose.{camera}.npy"
            if key in sample_camera:
                continue

            prev_idx = (target_idx // IndexScaleRatio) * IndexScaleRatio
            next_idx = prev_idx + IndexScaleRatio

            sample_camera[key] = interpolate_pose(
                sample_camera[f"{prev_idx:06d}.pose.{camera}.npy"],
                sample_camera[f"{next_idx:06d}.pose.{camera}.npy"],
                (target_idx - prev_idx) / IndexScaleRatio,
            ).astype(np.float32)

    # vehicle

    approx_motion = (
        sample_vehicle[f"{max_target:06d}.vehicle_pose.npy"]
        - sample_vehicle[f"{max_target-1:06d}.vehicle_pose.npy"]
    )

    approx_motion[:3, :3] = 0
    sample_vehicle[f"{max_target+1:06d}.vehicle_pose.npy"] = (
        sample_vehicle[f"{max_target:06d}.vehicle_pose.npy"]
        + approx_motion
    )

    sample_vehicle[f"{max_target+2:06d}.vehicle_pose.npy"] = (
        sample_vehicle[f"{max_target:06d}.vehicle_pose.npy"]
        + 2 * approx_motion
    )

    for camera in CameraNameMap.values():
        approx_motion = (
            sample_camera[f"{max_target:06d}.pose.{camera}.npy"]
            - sample_camera[f"{max_target-1:06d}.pose.{camera}.npy"]
        )

        approx_motion[:3, :3] = 0
        sample_camera[f"{max_target+1:06d}.pose.{camera}.npy"] = (
            sample_camera[f"{max_target:06d}.pose.{camera}.npy"]
            + approx_motion
        )

        sample_camera[f"{max_target+2:06d}.pose.{camera}.npy"] = (
            sample_camera[f"{max_target:06d}.pose.{camera}.npy"]
            + 2 * approx_motion
        )

    write_to_tar(sample_camera, output_root / "pose" / f"{clip_id}.tar")
    write_to_tar(sample_vehicle, output_root / "vehicle_pose" / f"{clip_id}.tar",)

# convert_bbox()
def convert_bbox(output_root: Path, clip_id: str, label_folder: Path, calib_folder: Path):
    sample = {"__key__": clip_id}
    label_files = sorted(label_folder.glob("*.txt"))
    calib_files = sorted(calib_folder.glob("*.pkl"))

    for frame_idx, (label_file, calib_file) in enumerate(zip(label_files, calib_files)):
        with open(calib_file, "rb") as f:
            calib = pickle.load(f)

        ego_to_world = calib["ego_to_world"].astype(np.float32)
        objects = {}

        with open(label_file) as f:
            lines = f.readlines()

        # first line is ego speed
        for line in lines[1:]:
            items = line.strip().split()
            cls = items[0]
            if cls not in OBJECT_MAP:
                continue

            x = float(items[1])
            y = float(items[2])
            z = float(items[3])
            l = float(items[4])
            w = float(items[5])
            h = float(items[6])
            yaw = float(items[7])
            vx = float(items[8])
            vy = float(items[9])
            obj_id = items[10]
            moving = np.sqrt(vx * vx + vy * vy) > 0.2
            T = np.eye(4, dtype=np.float32)

            T[:3, :3] = R.from_euler("z", yaw).as_matrix()
            T[:3, 3] = [x, y, z]

            object_to_world = ego_to_world @ T
            objects[obj_id] = {
                "object_to_world": object_to_world.tolist(),
                "object_lwh": [l, w, h],
                "object_type": OBJECT_MAP[cls],
                "object_is_moving": bool(moving),
            }

        sample[f"{frame_idx*IndexScaleRatio:06d}.all_object_info.json"] = objects

    write_to_tar(sample, output_root / "all_object_info" / f"{clip_id}.tar")

# convert_lidar()
def convert_lidar(output_root: Path, clip_id: str, lidar_folder: Path, calib_folder: Path):
    sample = {"__key__": clip_id}
    lidar_files = sorted(lidar_folder.glob("*.npz"))
    calib_files = sorted(calib_folder.glob("*.pkl"))

    for frame_idx, (lidar_file, calib_file) in enumerate(zip(lidar_files, calib_files)):
        with open(calib_file, "rb") as f:
            calib = pickle.load(f)

        lidar_to_ego = calib["lidar_to_ego"]
        ego_to_world = calib["ego_to_world"].astype(np.float32)
        lidar_to_world = ego_to_world @ lidar_to_ego

        lidar = np.load(lidar_file)["data"]
        xyz = lidar[:, :3]
        sample[
            f"{frame_idx*IndexScaleRatio:06d}.lidar_raw.npz"
        ] = encode_dict_to_npz_bytes(
            {"xyz": xyz.astype(np.float32), "lidar_to_world": lidar_to_world.astype(np.float32)}
        )

    write_to_tar(sample, output_root / "lidar_raw" / f"{clip_id}.tar")

# convert_image()
def convert_image(output_root: Path, clip_id: str, camera_root: Path, single_camera=False):
    for deep_name, rds_name in CameraNameMap.items():
        if single_camera and rds_name != "front":
            continue

        image_dir = camera_root / deep_name / clip_id
        image_files = sorted(image_dir.glob("*.jpg"))
        output_video = (output_root / f"pinhole_{rds_name}" / f"{clip_id}.mp4")
        output_video.parent.mkdir(parents=True, exist_ok=True)

        writer = imageio_v1.get_writer(
            output_video,
            fps=SourceFps,
            macro_block_size=None,
        )

        for img_file in image_files:
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            writer.append_data(img)
        writer.close()

# convert_timestamp()
def convert_timestamp(output_root: Path, clip_id: str, calib_folder: Path,):
    """
    Convert DeepAccident timestamps into RDS-HQ format.

    Since DeepAccident does not provide real timestamps,
    synthetic timestamps are generated assuming 10 Hz.

    Output:
        timestamp/
            clip_id.tar

    containing

        000000.timestamp_micros.txt
        000003.timestamp_micros.txt
        ...
    """
    sample = {"__key__": clip_id}
    calib_files = sorted(calib_folder.glob("*.pkl"))

    for frame_idx in range(len(calib_files)):
        target_idx = frame_idx * IndexScaleRatio
        timestamp = int(frame_idx * TimestampInterval)
        sample[f"{target_idx:06d}.timestamp_micros.txt"] = str(timestamp)

    write_to_tar(sample, output_root / "timestamp" / f"{clip_id}.tar")

# convert_one_clip()
def convert_one_clip(clip_id, output_root, calib_root, image_root, label_root, lidar_root, single_camera=False):
    calib_folder = calib_root / clip_id
    label_folder = label_root / clip_id
    lidar_folder = lidar_root / clip_id
    image_folder = image_root

    convert_intrinsics(output_root, clip_id, calib_folder, image_folder)
    convert_pose(output_root, clip_id, calib_folder)
    convert_bbox(output_root, clip_id, label_folder, calib_folder)
    convert_lidar(output_root, clip_id, lidar_folder, calib_folder)
    convert_image(output_root, clip_id, image_folder, single_camera)
    convert_timestamp(output_root, clip_id, calib_folder)

@click.command()
@click.option("--input_root", "-i", type=click.Path(exists=True))
@click.option("--output_root", "-o", type=click.Path())
@click.option("--num_workers", "-n", default=1)
@click.option("--single_camera", "-s", is_flag=True)
def main(input_root, output_root, num_workers, single_camera):
    input_root = Path(input_root)
    output_root = Path(output_root)
    platforms = [
        "ego_vehicle",
        "ego_vehicle_behind",
        "other_vehicle",
        "other_vehicle_behind",
        "infrastructure",
    ]

    for platform in platforms:
        platform_root = input_root / platform
        if not platform_root.exists():
            print(f"Skip {platform}: not found.")
            continue

        print(f"\nProcessing {platform}")
        calib_root = platform_root / "calib"
        label_root = platform_root / "label"
        lidar_root = platform_root / "lidar01"
        image_root = platform_root  # image root is the platform root itself
        clip_ids = sorted(
            [p.stem for p in calib_root.iterdir() if p.is_dir()]
        )

        worker = partial(
            convert_one_clip,
            output_root=output_root / platform,
            calib_root=calib_root,
            image_root=image_root,
            label_root=label_root,
            lidar_root=lidar_root,
            single_camera=single_camera,
        )

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(
                tqdm(
                    executor.map(worker, clip_ids),
                    total=len(clip_ids),
                    desc=platform,
                )
            )

if __name__ == "__main__":
    main()