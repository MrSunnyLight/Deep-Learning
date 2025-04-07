cd ../

#python train_virtual_softmax.py --dataset cifar100 --approach sce --batch_size 128 --epochs 200 --dir runs_virtual_softmax/100_x_norm --loop_num 0
#python train_virtual_softmax.py --dataset cifar100 --approach sce --batch_size 128 --epochs 200 --dir runs_virtual_softmax/100_x_norm --loop_num 1
#python train_virtual_softmax.py --dataset cifar100 --approach sce --batch_size 128 --epochs 200 --dir runs_virtual_softmax/100_x_norm --loop_num 2
#python train_virtual_softmax.py --dataset cifar100 --approach sce --batch_size 128 --epochs 200 --dir runs_virtual_softmax/100_x_norm --loop_num 3
#python train_virtual_softmax.py --dataset cifar100 --approach sce --batch_size 128 --epochs 200 --dir runs_virtual_softmax/100_x_norm --loop_num 4
#
#python train_virtual_softmax.py --dataset cifar10 --approach sce --batch_size 128 --epochs 200 --dir runs_virtual_softmax/10_x_norm --loop_num 0
#python train_virtual_softmax.py --dataset cifar10 --approach sce --batch_size 128 --epochs 200 --dir runs_virtual_softmax/10_x_norm --loop_num 1
#python train_virtual_softmax.py --dataset cifar10 --approach sce --batch_size 128 --epochs 200 --dir runs_virtual_softmax/10_x_norm --loop_num 2
#python train_virtual_softmax.py --dataset cifar10 --approach sce --batch_size 128 --epochs 200 --dir runs_virtual_softmax/10_x_norm --loop_num 3
#python train_virtual_softmax.py --dataset cifar10 --approach sce --batch_size 128 --epochs 200 --dir runs_virtual_softmax/10_x_norm --loop_num 4

python train.py --dataset CUB200-2011 --data_root ../datasets/CUB_200_2011 --approach virtual_softmax --batch_size 8 --epochs 200 --dir runs/virtual_softmax/CUB --loop_num 0
python train.py --dataset CUB200-2011 --data_root ../datasets/CUB_200_2011 --approach virtual_softmax --batch_size 8 --epochs 200 --dir runs/virtual_softmax/CUB --loop_num 1
python train.py --dataset CUB200-2011 --data_root ../datasets/CUB_200_2011 --approach virtual_softmax --batch_size 8 --epochs 200 --dir runs/virtual_softmax/CUB --loop_num 2
