import click
from pathlib import Path


WEATHER_MAP = {
    "ClearNoon": "clear daytime",
    "WetNoon": "wet daytime",
    "CloudyNoon": "cloudy daytime",
    "MidRainyNoon": "moderate rain during daytime",
    "HardRainNoon": "heavy rain during daytime",
    "ClearSunset": "clear sunset",
    "CloudySunset": "cloudy sunset",
    "WetSunset": "wet sunset",
    "MidRainSunset": "moderate rain at sunset",
    "HardRainSunset": "heavy rain at sunset",
    "ClearNight": "clear night",
    "WetNight": "wet night",
    "MidRainNight": "moderate rain at night",
    "HardRainNight": "heavy rain at night",
}

LIDAR_CATEGORIES = {
    "ego_vehicle", "ego_vehicle_behind",
    "other_vehicle",  "other_vehicle_behind",
    "infrastructure"
}

LIDAR_VIEW = "ftheta_front"


def clean_value(v):
    return v.strip().replace("_", " ")


def parse_meta(meta_file: Path):
    """
    Parse one DeepAccident meta txt file.
    """

    info = {}

    with open(meta_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines):
        weather = lines[0].split()[0]
        info["weather"] = weather

    for line in lines[1:]:

        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        info[key.strip()] = value.strip()

    return info


def find_matching_videos(base_name: str, lidar_dir: Path):

    matches = []

    for category in LIDAR_CATEGORIES:
        pattern = f"{category}__{base_name}_*.mp4"
        matches.extend(sorted(lidar_dir.glob(pattern)))

    return matches


def build_caption(info):

    weather = WEATHER_MAP.get(
        info.get("weather", ""),
        clean_value(info.get("weather", "unknown weather"))
    )

    road = clean_value(info.get("road_type", "road"))

    caption = []

    caption.append(
        f"A driving scene during {weather} on a {road}."
    )

    ego_dir = info.get("ego_vehicle_direction")
    if ego_dir:
        caption.append(
            f"The ego vehicle is driving {clean_value(ego_dir)}."
        )

    other_dir = info.get("other_vehicle_direction")
    spawn = info.get("another_vehicle_spawn_side")

    if other_dir and spawn:
        caption.append(
            f"Another vehicle approaches from the {clean_value(spawn)} and drives {clean_value(other_dir)}."
        )
    elif other_dir:
        caption.append(
            f"Another vehicle is driving {clean_value(other_dir)}."
        )

    collision = info.get("colliding agents")
    if collision:

        if collision.lower() != "none none":
            caption.append(
                f"A collision involves {clean_value(collision)}."
            )
        else:
            caption.append(
                "No collision occurs during the scene."
            )

    agent_ids = info.get("agents id")
    if agent_ids:
        n = len(agent_ids.split())
        caption.append(
            f"There are {n} tracked traffic participants in the scene."
        )

    caption.append(
        "The scene is captured from the ego vehicle perspective."
    )

    return " ".join(caption)


@click.command()
@click.option(
    "--input_dir",
    "-i",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(),
    required=True,
)
@click.option(
    "--lidar_dir",
    type=click.Path(exists=True),
    required=True,
)
def main(input_dir, output_dir, lidar_dir):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    lidar_dir = Path(lidar_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_files = sorted(input_dir.glob("*.txt"))
    print(f"Found {len(meta_files)} metadata files.")

    for meta_file in meta_files:
        info = parse_meta(meta_file)
        caption = build_caption(info)
        base_name = meta_file.stem
        video_files = find_matching_videos(base_name, lidar_dir)

        if not video_files:
            print("No matching LiDAR videos found for {base_name}")
            continue

        for video_file in video_files:
            output_file = output_dir / f"{video_file.stem}.txt"
            with open(output_file, "w") as f:
                f.write(caption)
            print("Saved {output_file.name}")


if __name__ == "__main__":
    main()