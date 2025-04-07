cd ../

python train.py --network cnn --dataset mnist --approach msce_virtual_learning_strategy --loss_type Focal --learning_rate 0.01 --batch_size 128 --epochs 100 --dir runs/msce_virtual_learning_strategy/mnist --loop_num 0 --s 2
python train.py --network cnn --dataset mnist --approach msce_virtual_learning_strategy --loss_type Focal --learning_rate 0.01 --batch_size 128 --epochs 100 --dir runs/msce_virtual_learning_strategy/mnist --loop_num 1 --s 2
python train.py --network cnn --dataset mnist --approach msce_virtual_learning_strategy --loss_type Focal --learning_rate 0.01 --batch_size 128 --epochs 100 --dir runs/msce_virtual_learning_strategy/mnist --loop_num 2 --s 2

python train.py --dataset cifar10 --approach msce_virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/msce_virtual_learning_strategy/10 --loop_num 0 --s 2
python train.py --dataset cifar10 --approach msce_virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/msce_virtual_learning_strategy/10 --loop_num 1 --s 2
python train.py --dataset cifar10 --approach msce_virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/msce_virtual_learning_strategy/10 --loop_num 2 --s 2

python train.py --dataset cifar100 --approach msce_virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/msce_virtual_learning_strategy/100 --loop_num 0 --s 2
python train.py --dataset cifar100 --approach msce_virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/msce_virtual_learning_strategy/100 --loop_num 1 --s 2
python train.py --dataset cifar100 --approach msce_virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/msce_virtual_learning_strategy/100 --loop_num 2 --s 2
