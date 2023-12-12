cd /home/fourier/paulo/sam-hq/train
date > start_date.txt
nohup python -m torch.distributed.launch --nproc_per_node=1 train.py --checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b --output work_dirs/hq_sam_b_001_300 --local_rank 0 --batch_size_train 2 > train_output.txt 2>&1 &