cd ../

python train.py --dataset cifar10 --approach msce_resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/msce_resultant_virtual_focal/10 --loop_num 0
python train.py --dataset cifar10 --approach msce_resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/msce_resultant_virtual_focal/10 --loop_num 1
python train.py --dataset cifar10 --approach msce_resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/msce_resultant_virtual_focal/10 --loop_num 2

python train.py --dataset cifar100 --approach msce_resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/msce_resultant_virtual_focal/100 --loop_num 0
python train.py --dataset cifar100 --approach msce_resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/msce_resultant_virtual_focal/100 --loop_num 1
python train.py --dataset cifar100 --approach msce_resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/msce_resultant_virtual_focal/100 --loop_num 2
