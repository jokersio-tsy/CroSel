# CroSel: Cross Selection of Confident Pseudo Labels for Partial-Label Learning

This is official implication for our paper: 
CroSel: Cross Selection of Confident Pseudo Labels for Partial-Label Learning (Accepted by CVPR 2024) 

[[Arxiv]](https://arxiv.org/abs/2303.10365)

## Algorithm overview
![image-frame](img\frame.png)
![image-pseudo_code](img\pseudo_code.png)

## Train Command

```bash
#train_command for cifar10 q=0.5

python mix_main.py --dataset cifar10  --arch wideresnet --batch-size 64 --lr 0.1 \
--seed 5 --out cifar10@q05 --partial_rate 0.5 --epochs 200 --gpu_id 7 --sharpen_T 0.5 --use_mix --lambda_cr 4

#tmp_command for cifar100 q=0.1

python mix_main.py --dataset cifar100  --arch wideresnet --batch-size 64 --lr 0.1 \
--seed 5 --out cifar100@q01--partial_rate 0.10 --epochs 200 --gpu_id 6 --sharpen_T 0.5 --use_mix
```

