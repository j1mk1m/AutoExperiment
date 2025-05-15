from pathlib import Path


def get_5k_images(original_folder: Path, new_folder: Path):
    """The new folder will contain ~5000 images that will NOT be copied (just symlinked)"""
    # 63 * 80 = 5040
    n = 63

    if new_folder.exists():
        print(f"{new_folder} already exists, skipping...")
        return

    files_to_keep = []
    for attribute in original_folder.glob("*"):
        for category in attribute.glob("*"):
            files = sorted(category.glob("**/*.png"))
            if len(files) != 104:
                print(attribute.name, category.name, len(files))
            files_to_keep.extend(files[-n:])

    print(f"Keeping {len(files_to_keep)} images from {original_folder} in {new_folder}")
    new_folder.mkdir(parents=True, exist_ok=True)
    for file in files_to_keep:
        output_filename = (
            file.parent.parent.name + "_" + file.parent.name + "_" + file.name
        )
        output_file = new_folder / output_filename
        output_file.symlink_to(file.absolute())


if __name__ == "__main__":
    # Get only 5K images per folder to make results comparable with the original paper
    folder_pairs = [
        (
            Path("results/celeba_single/iti_gen"),
            Path("results/celeba_single/5K_iti_gen"),
        ),
        (Path("results/celeba_single/hps"), Path("results/celeba_single/5K_hps")),
        (
            Path("results/celeba_single/hps_negative"),
            Path("results/celeba_single/5K_hps_negative"),
        ),
    ]

    for original_folder, new_folder in folder_pairs:
        get_5k_images(original_folder, new_folder)
