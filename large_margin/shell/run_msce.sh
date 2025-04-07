cd ../

python train.py --dataset cifar10 --approach msce --batch_size 128 --epochs 200 --dir runs_msce/10 --loop_num 0
python train.py --dataset cifar10 --approach msce --batch_size 128 --epochs 200 --dir runs_msce/10 --loop_num 1
python train.py --dataset cifar10 --approach msce --batch_size 128 --epochs 200 --dir runs_msce/10 --loop_num 2

python train.py --dataset cifar100 --approach msce --batch_size 128 --epochs 200 --dir runs_msce/100 --loop_num 0
python train.py --dataset cifar100 --approach msce --batch_size 128 --epochs 200 --dir runs_msce/100 --loop_num 1
python train.py --dataset cifar100 --approach msce --batch_size 128 --epochs 200 --dir runs_msce/100 --loop_num 2
