cd ../

python train.py --dataset cifar10 --approach virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/virtual_learning_strategy_useat100/10 --loop_num 0
python train.py --dataset cifar10 --approach virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/virtual_learning_strategy_useat100/10 --loop_num 1
python train.py --dataset cifar10 --approach virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/virtual_learning_strategy_useat100/10 --loop_num 2

python train.py --dataset cifar100 --approach virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/virtual_learning_strategy_useat100/100 --loop_num 0
python train.py --dataset cifar100 --approach virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/virtual_learning_strategy_useat100/100 --loop_num 1
python train.py --dataset cifar100 --approach virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/virtual_learning_strategy_useat100/100 --loop_num 2
