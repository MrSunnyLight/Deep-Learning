cd ../

python train.py --dataset cifar10 --approach sce --loss_type LMSoftmax --batch_size 128 --epochs 200 --dir runs_LMSoftmax/10 --loop_num 0
python train.py --dataset cifar10 --approach sce --loss_type LMSoftmax --batch_size 128 --epochs 200 --dir runs_LMSoftmax/10 --loop_num 1
python train.py --dataset cifar10 --approach sce --loss_type LMSoftmax --batch_size 128 --epochs 200 --dir runs_LMSoftmax/10 --loop_num 2

python train.py --dataset cifar100 --approach sce --loss_type LMSoftmax --batch_size 128 --epochs 200 --dir runs_LMSoftmax/100 --loop_num 0
python train.py --dataset cifar100 --approach sce --loss_type LMSoftmax --batch_size 128 --epochs 200 --dir runs_LMSoftmax/100 --loop_num 1
python train.py --dataset cifar100 --approach sce --loss_type LMSoftmax --batch_size 128 --epochs 200 --dir runs_LMSoftmax/100 --loop_num 2
