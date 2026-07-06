import shutil
from pathlib import Path
import click
from tqdm import tqdm

SUBFOLDERS = [
    "all_object_info",
    "lidar_raw",
    "pinhole_front",
    "pinhole_front_left",
    "pinhole_front_right",
    "pinhole_back",
    "pinhole_back_left",
    "pinhole_back_right",
    "pinhole_intrinsic",
    "pose",
    "vehicle_pose",
    "timestamp",
]

PLATFORMS = [
    "ego_vehicle",
    "ego_vehicle_behind",
    "other_vehicle",
    "other_vehicle_behind",
    "infrastructure",
]


@click.command()
@click.option(
    "--input_root",
    "-i",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--output_root",
    "-o",
    type=click.Path(),
    required=True,
)
@click.option(
    "--copy",
    is_flag=True,
    help="Copy instead of move.",
)
def main(input_root, output_root, copy):
    input_root = Path(input_root)
    output_root = Path(output_root)

    for folder in SUBFOLDERS:
        (output_root / folder).mkdir(parents=True, exist_ok=True)

    for platform in PLATFORMS:
        platform_root = input_root / platform
        if not platform_root.exists():
            continue
        print(f"\nProcessing {platform}")

        for folder in SUBFOLDERS:
            src_folder = platform_root / folder
            if not src_folder.exists():
                continue

            files = sorted(src_folder.iterdir())
            for src in tqdm(files, desc=f"{platform}/{folder}"):
                if src.suffix == ".mp4":
                    new_name = f"{platform}__{src.stem}.mp4"
                elif src.suffix == ".tar":
                    new_name = f"{platform}__{src.stem}.tar"
                else:
                    new_name = f"{platform}__{src.name}"
                dst = output_root / folder / new_name

                if dst.exists():
                    print(f"Skip existing {dst.name}")
                    continue

                if copy:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)

    print("\nFinished.")


if __name__ == "__main__":
    main()