import torch

import argparse

parser = argparse.ArgumentParser(
    description="Merge sam weights with sam-hq mask decoder weights."
)

parser.add_argument(
    "--sam", type=str, required=True, help="The path to the SAM model checkpoint."
)

parser.add_argument(
    "--hq", type=str, required=True, help="The path to the SAM model checkpoint."
)

parser.add_argument(
    "--output", type=str, required=True, help="The filename to save the ONNX model to."
)


if __name__ == "__main__":
    args = parser.parse_args()
    sam_ckpt = torch.load(args.sam, map_location=torch.device('cpu'))
    hq_decoder = torch.load(args.hq, map_location=torch.device('cpu'))

    for key in hq_decoder.keys():
        sam_key = 'mask_decoder.'+ key
        if sam_key not in sam_ckpt.keys():
            sam_ckpt[sam_key] = hq_decoder[key]

    torch.save(sam_ckpt, args.output)