#!/bin/bash

url="https://open.feishu.cn/open-apis/bot/v2/hook/e4ca6d5a-a6e7-4529-bff0-d94a6d2d7196"
header="Content-Type: application/json"
function send_text(){
    textmsg='{"msg_type":"text","content":{"text":"'$t'"}}'
    curl -s -H $header -d "$textmsg" "$url"
}

# log=/media/disk/02drive/05yueye/code/PDV/logs/train-$(date +%Y-%m-%d_%H-%M-%S).log

t="Dione test program $$ started!"
send_text
# /home/yueye/anaconda3/envs/pdv36/bin/
# CUDA_VISIBLE_DEVICES=1 /media/disk/02drive/05yueye/anaconda3/envs/pdv/bin/python /media/disk/02drive/05yueye/code/PDV/tools/test.py \
#                                                         --cfg_file ./cfgs/wanji_models/pdv.yaml \
#                                                         --batch_size 8 \
#                                                         --workers 8 \
#                                                         --ckpt ../output/wanji_models/pdv/default/ckpt/checkpoint_epoch_36.pth

CUDA_VISIBLE_DEVICES=3 /media/disk/02drive/05yueye/anaconda3/envs/pdv/bin/python /media/disk/02drive/05yueye/code/PDV/tools/test.py \
                                                        --eval_all True \
                                                        --ckpt_dir ../output/wanji_models/pdv/default/ckpt \
                                                        --cfg_file cfgs/wanji_models/pdv.yaml \
                                                        --batch_size 8 \
                                                        --workers 8
# /media/disk/02drive/05yueye/code/PDV/output/wanji_models/pdv/default/ckpt
t="Dione test program $$ finished!"
send_text
