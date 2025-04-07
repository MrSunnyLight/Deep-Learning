cd ../

python train.py --dataset cifar10 --approach virtual_softmax_rsm --batch_size 128 --epochs 200 --dir runs_virtual_softmax_rsm/10 --loop_num 0
python train.py --dataset cifar10 --approach virtual_softmax_rsm --batch_size 128 --epochs 200 --dir runs_virtual_softmax_rsm/10 --loop_num 1
python train.py --dataset cifar10 --approach virtual_softmax_rsm --batch_size 128 --epochs 200 --dir runs_virtual_softmax_rsm/10 --loop_num 2

python train.py --dataset cifar100 --approach virtual_softmax_rsm --batch_size 128 --epochs 200 --dir runs_virtual_softmax_rsm/100 --loop_num 0
python train.py --dataset cifar100 --approach virtual_softmax_rsm --batch_size 128 --epochs 200 --dir runs_virtual_softmax_rsm/100 --loop_num 1
python train.py --dataset cifar100 --approach virtual_softmax_rsm --batch_size 128 --epochs 200 --dir runs_virtual_softmax_rsm/100 --loop_num 2
