import argparse
import torch
from cleanfid import fid


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"

    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass

    return "cpu"


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", type=str, required=True)

    mutex_group = parser.add_mutually_exclusive_group(required=True)
    mutex_group.add_argument("--folder2", type=str)
    mutex_group.add_argument("--ffhq", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    device = get_device()
    args = get_parameters()

    if args.folder2:
        kwargs = {"fdir2": args.folder2}
    elif args.ffhq:
        kwargs = {
            "dataset_name": "FFHQ",
            "dataset_res": 1024,
            "dataset_split": "trainval70k",
        }
    else:
        raise Exception("--folder2 or --ffhq must be specified!")

    score = fid.compute_fid(args.folder1, device=device, num_workers=0, **kwargs)
    print(f"FID score for {args.folder1}: {score:.3f}")
