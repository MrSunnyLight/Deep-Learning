cd ../

python train.py --dataset cifar10 --approach msce_LMSoftmax --loss_type LMSoftmax --batch_size 128 --epochs 200 --dir runs_msce_LMSoftmax/10_s_64 --loop_num 0
python train.py --dataset cifar10 --approach msce_LMSoftmax --loss_type LMSoftmax --batch_size 128 --epochs 200 --dir runs_msce_LMSoftmax/10_s_64 --loop_num 1
python train.py --dataset cifar10 --approach msce_LMSoftmax --loss_type LMSoftmax --batch_size 128 --epochs 200 --dir runs_msce_LMSoftmax/10_s_64 --loop_num 2

python train.py --dataset cifar100 --approach msce_LMSoftmax --loss_type LMSoftmax --batch_size 128 --epochs 200 --dir runs_msce_LMSoftmax/100_s_64 --loop_num 0
python train.py --dataset cifar100 --approach msce_LMSoftmax --loss_type LMSoftmax --batch_size 128 --epochs 200 --dir runs_msce_LMSoftmax/100_s_64 --loop_num 1
python train.py --dataset cifar100 --approach msce_LMSoftmax --loss_type LMSoftmax --batch_size 128 --epochs 200 --dir runs_msce_LMSoftmax/100_s_64 --loop_num 2
