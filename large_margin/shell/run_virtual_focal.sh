cd ../

python train.py --dataset cifar10 --approach virtual_focal --loss_type Focal --batch_size 128 --epochs 200 --dir runs_virtual_focal/10_gamma_2 --loop_num 0
python train.py --dataset cifar10 --approach virtual_focal --loss_type Focal --batch_size 128 --epochs 200 --dir runs_virtual_focal/10_gamma_2 --loop_num 1
python train.py --dataset cifar10 --approach virtual_focal --loss_type Focal --batch_size 128 --epochs 200 --dir runs_virtual_focal/10_gamma_2 --loop_num 2

python train.py --dataset cifar100 --approach virtual_focal --loss_type Focal --batch_size 128 --epochs 200 --dir runs_virtual_focal/100_gamma_2 --loop_num 0
python train.py --dataset cifar100 --approach virtual_focal --loss_type Focal --batch_size 128 --epochs 200 --dir runs_virtual_focal/100_gamma_2 --loop_num 1
python train.py --dataset cifar100 --approach virtual_focal --loss_type Focal --batch_size 128 --epochs 200 --dir runs_virtual_focal/100_gamma_2 --loop_num 2
