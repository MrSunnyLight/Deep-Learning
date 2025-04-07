cd ../

python train.py --dataset cifar10 --approach resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/resultant_virtual_focal/10 --loop_num 3
python train.py --dataset cifar10 --approach resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/resultant_virtual_focal/10 --loop_num 4
python train.py --dataset cifar10 --approach resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/resultant_virtual_focal/10 --loop_num 5

#python train.py --dataset cifar100 --approach resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/resultant_virtual_focal/100_s_2 --loop_num 0 --s 2
#python train.py --dataset cifar100 --approach resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/resultant_virtual_focal/100_s_2 --loop_num 1 --s 2
#python train.py --dataset cifar100 --approach resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/resultant_virtual_focal/100_s_2 --loop_num 2 --s 2
