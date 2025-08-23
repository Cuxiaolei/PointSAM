accelerate launch --mixed_precision=bf16 train.py --config large --test

python train.py --config large +test=True ++test_ckpt_path=./logs/large_default/checkpoints/checkpoint_19