#!/bin/bash

url="https://open.feishu.cn/open-apis/bot/v2/hook/e4ca6d5a-a6e7-4529-bff0-d94a6d2d7196"
header="Content-Type: application/json"
function send_text(){
    textmsg='{"msg_type":"text","content":{"text":"'$t'"}}'
    curl -s -H $header -d "$textmsg" "$url"
}

# log=/media/disk/02drive/05yueye/code/PDV/logs/train-$(date +%Y-%m-%d_%H-%M-%S).log

t="Dione program $$ started!"
send_text
# /home/yueye/anaconda3/envs/pdv36/bin/
# CUDA_VISIBLE_DEVICES=3 /media/disk/02drive/05yueye/anaconda3/envs/pdv/bin/python /media/disk/02drive/05yueye/code/PDV/tools/train_wanji.py \
#                                                         --cfg_file cfgs/wanji_models/pdv.yaml \
#                                                         --batch_size 1 \
#                                                         --epochs 1 \
#                                                         --workers 1 \
#                                                         --ckpt_save_interval 1

CUDA_VISIBLE_DEVICES=3 /media/disk/02drive/05yueye/anaconda3/envs/pdv/bin/python /media/disk/02drive/05yueye/code/PDV/tools/train.py \
                                                        --cfg_file cfgs/kitti_models/pdv.yaml \
                                                        --batch_size 2 \
                                                        --epochs 1 \
                                                        --ckpt_save_interval 1

t="Dione program $$ finished!"
send_text