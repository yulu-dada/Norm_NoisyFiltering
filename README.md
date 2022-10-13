# Our code is implemented on Fairseq.

This is for our paper "Norm-based Noisy Corpora Filtering and Refurbishing in Neural Machine Translation" published at EMNLP2022.

## env: Fairseq-0.9, Pytorch-1.6
## code structure:
- mycode : Noisy Corpora Filtering
- mycode-kd: Noisy Label Refurbishing

## Step 1: Preprocess the data using rule-based filtering (please refer to appendix C)

## Step 2: Train
sent-threshold: k in equation(9)
```python
SRC=$1
direct=$SRC-zh
data_bin=../program/zh-$SRC/data-bin-train/
model_dir=./models/$direct-model
usr_dir=./mycode
mkdir -p $model_dir

export CUDA_VISIBLE_DEVICES=0,1
nohup fairseq-train $data_bin \
    --user-dir $usr_dir --arch my_arch --criterion my_label_smoothed_cross_entropy --task my_translation_task --report-accuracy \
    --source-lang $SRC --target-lang zh \
    --sent-threshold 2.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0007 --min-lr 1e-09 \
    --weight-decay 0.0 --label-smoothing 0.1 \
    --max-tokens 8000 --update-freq 1 --no-progress-bar --max-update 150000 \
    --log-interval 100 --save-interval-updates 1000 --keep-interval-updates 10 --save-interval 10000 \
    --ddp-backend=no_c10d \
    --encoder-normalize-before --decoder-normalize-before \
    --save-dir $model_dir > log.train-$direct &
```

The following setting is very important to ensure the stability of our proposed metric!!!
```--encoder-normalize-before --decoder-normalize-before```

## Step 3: Inference
```python
SRC=$1
direct=$SRC-zh
model_name=$direct-model
test_path=../program/zh-$SRC/data-bin-test/
model_dir=./models/$model_name
output=./outputs/$model_name
mkdir -p $output
usr_dir=./mycode

python ../../../fairseq/scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/checkpoint_average.pt --num-update-checkpoints 10

export CUDA_VISIBLE_DEVICES=0
nohup fairseq-generate $test_path/ \
    --max-tokens 23000 \
    --source-lang $SRC --target-lang zh \
    --gen-subset valid \
    --path $model_dir/checkpoint_average.pt \
    --beam 5 --lenpen 1.2 \
    --remove-bpe=sentencepiece \
    --user-dir $usr_dir > $output/$direct.txt &
```
