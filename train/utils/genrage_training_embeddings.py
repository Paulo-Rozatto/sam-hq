import argparse
from dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from misc import init_distributed_mode

ft_001_300 = {"name": "ft_001_300",
              "im_dir": "train/data/fine-tune-001-300/train_images",
              "gt_dir": "train/data/fine-tune-001-300/train_masks",
              "im_ext": ".jpg",
              "gt_ext": ".jpg"}


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hflip", default=True,
                        help="Use horizontal flip transformation")
    parser.add_argument("--jitter", default=True,
                        help="Use Large Scale Jitter")
    parser.add_argument("--batch_size", default=1, type=int,)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    # parser.add_argument('--local-rank', type=int, help='local rank for dist')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')

    return parser.parse_args()


def get_transforms(args):
    transforms = []
    if args.hflip:
        transforms.append(RandomHFlip())

    if args.jitter:
        transforms.append(LargeScaleJitter())
    else:
        transforms.append(Resize())


if __name__ == "__main__":
    args = get_args_parser()

    init_distributed_mode(args)

    datasets = [ft_001_300]

    print("--- Creating dataloader ---")

    image_list = get_im_gt_name_dict(datasets, flag="train")

    data_loaders, datasets = create_dataloaders(
        image_list,
        my_transforms=get_transforms(args),
        batch_size=args.batch_size,
        training=True,
        distributed=False
    )
    print("->", len(data_loaders), " dataloaders created")
