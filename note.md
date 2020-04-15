pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


mkdir -p output/squadv2

SQUAD_DIR=data/squad
#INIT_CKPT_DIR=xlnet_cased_L-24_H-1024_A-16
INIT_CKPT_DIR=xlnet_cased_L-12_H-768_A-12

python run_squad.py \
  --use_tpu=False \
  --do_prepro=True \
  --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
  --train_file=${SQUAD_DIR}/train-v2.0.json \
  --output_dir=output/squadv2 \
  --uncased=False \
  --max_seq_length=512 \
  

2020-04-15 20:16:45
```
(env) xuehp@haomeiya001:/home/xuehp/git/xlnet$

nohup python run_squad.py   --use_tpu=False   --do_prepro=True   --spiece_model_file=${INIT_CKPT_DIR}/spiece.model   --train_file=${SQUAD_DIR}/train-v2.0.json   --output_dir=output/squadv2   --uncased=False   --max_seq_length=512  > log.4.15.txt &
```