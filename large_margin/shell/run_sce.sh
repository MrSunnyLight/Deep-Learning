cd ../

#python train.py --dataset cifar10 --approach sce --batch_size 128 --epochs 200 --dir runs_sce/10 --loop_num 0
#python train.py --dataset cifar10 --approach sce --batch_size 128 --epochs 200 --dir runs_sce/10 --loop_num 1
#python train.py --dataset cifar10 --approach sce --batch_size 128 --epochs 200 --dir runs_sce/10 --loop_num 2
#
#python train.py --dataset cifar100 --approach sce --batch_size 128 --epochs 200 --dir runs_sce/10 --loop_num 0
#python train.py --dataset cifar100 --approach sce --batch_size 128 --epochs 200 --dir runs_sce/10 --loop_num 1
#python train.py --dataset cifar100 --approach sce --batch_size 128 --epochs 200 --dir runs_sce/10 --loop_num 2

python train.py --dataset SVHN --approach sce --batch_size 128 --epochs 200 --dir runs/sce/SVHN_ --loop_num 0
python train.py --dataset SVHN --approach sce --batch_size 128 --epochs 200 --dir runs/sce/SVHN_ --loop_num 1
python train.py --dataset SVHN --approach sce --batch_size 128 --epochs 200 --dir runs/sce/SVHN_ --loop_num 2

#python train.py --dataset CUB200-2011 --data_root ../datasets/CUB_200_2011 --approach sce --batch_size 8 --epochs 200 --dir runs/sce/CUB --loop_num 0
#python train.py --dataset CUB200-2011 --data_root ../datasets/CUB_200_2011 --approach sce --batch_size 8 --epochs 200 --dir runs/sce/CUB --loop_num 1
#python train.py --dataset CUB200-2011 --data_root ../datasets/CUB_200_2011 --approach sce --batch_size 8 --epochs 200 --dir runs/sce/CUB --loop_num 2
