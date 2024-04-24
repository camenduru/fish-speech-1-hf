#!/bin/bash
set -e

mkdir -p checkpoints

if [ -e checkpoints/text2semantic-medium-v1-2k.pth ]; then
    echo "checkpoints/text2semantic-medium-v1-2k.pth already exists"
else
    echo "Downloading text2semantic-medium-v1-2k.pth"
    wget -O checkpoints/text2semantic-medium-v1-2k.pth $CKPT_SEMANTIC
fi

if [ -e checkpoints/vq-gan-group-fsq-2x1024.pth ]; then
    echo "checkpoints/vq-gan-group-fsq-2x1024.pth already exists"
else
    echo "Downloading vq-gan-group-fsq-2x1024.pth"
    wget -O checkpoints/vq-gan-group-fsq-2x1024.pth $CKPT_VQGAN
fi
