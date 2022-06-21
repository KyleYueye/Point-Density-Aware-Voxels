#!/bin/bash

url="https://open.feishu.cn/open-apis/bot/v2/hook/e4ca6d5a-a6e7-4529-bff0-d94a6d2d7196"
header="Content-Type: application/json"
function send_text(){
    textmsg='{"msg_type":"text","content":{"text":"'$t'"}}'
    curl -s -H $header -d "$textmsg" "$url"
}

# log=/media/disk/02drive/05yueye/code/PDV/logs/demo-$(date +%Y-%m-%d_%H-%M-%S).log

t="Dione program $$ started!"
# send_text
# /home/yueye/anaconda3/envs/pdv36/bin/
CUDA_VISIBLE_DEVICES=1 /media/disk/02drive/05yueye/anaconda3/envs/pdv/bin/python /media/disk/02drive/05yueye/code/PDV/tools/demo.py \
                                     --cfg_file cfgs/wanji_models/pdv.yaml \
                                    --ckpt ../output/wanji_models/pdv/default/ckpt/checkpoint_epoch_34.pth \
                                    --data_path /media/disk/02drive/data/20211215161104/demo \
                                    --ext .pcd
                                    # 3850 4050 4130
# ../output/wanji_models/pdv/default/ckpt/checkpoint_epoch_20.pth \
# ../output/previous/ckpt/checkpoint_epoch_20.pth \
# CUDA_VISIBLE_DEVICES=3 /media/disk/02drive/05yueye/anaconda3/envs/pdv/bin/python /media/disk/02drive/05yueye/code/PDV/tools/demo.py \
#                                      --cfg_file cfgs/kitti_models/pdv.yaml \
#                                     --ckpt /media/disk/02drive/05yueye/code/PDV/model/pdv_kitti.pth \
#                                     --data_path /media/disk/02drive/Kitti_dataset/kitti/training/velodyne/000010.bin \
#                                     --ext .bin

# /media/disk/02drive/05yueye/code/pcd2bin/bin/00_00000.bin
# /media/disk/02drive/Kitti_dataset/kitti/testing/velodyne/000008.bin
# /media/disk/02drive/05yueye/code/pcd2bin/bin/20200808090634_00002.bin \
# tensorboard --logdir=/media/disk/02drive/05yueye/code/PDV/output/wanji_models/pdv/default/tensorboard
t="Dione program $$ finished!"
# send_text
