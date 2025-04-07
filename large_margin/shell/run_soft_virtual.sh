# cd ../

venv/Scripts/python.exe train.py --device cpu --network resnet18 --dataset cifar100 --approach largest_virtual --batch_size 128 --epochs 1 --select 8 --scale 0.6  --dir runs/largest_virtual/06_16 --loop_num 0
# python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 3 --dir runs/largest_virtual/10_3 --loop_num 1
# python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 3 --dir runs/largest_virtual/10_3 --loop_num 2

#python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 4 --dir runs/largest_virtual/10_4 --loop_num 0
#python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 4 --dir runs/largest_virtual/10_4 --loop_num 1
#python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 4 --dir runs/largest_virtual/10_4 --loop_num 2
#
#python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 5 --dir runs/largest_virtual/10_5 --loop_num 2
#python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 5 --dir runs/largest_virtual/10_5 --loop_num 3
#python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 5 --dir runs/largest_virtual/10_5 --loop_num 4
#
#python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 6 --dir runs/largest_virtual/10_6 --loop_num 0
#python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 6 --dir runs/largest_virtual/10_6 --loop_num 1
#python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 6 --dir runs/largest_virtual/10_6 --loop_num 2
#
#python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 7 --dir runs/largest_virtual/10_7 --loop_num 0
#python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 7 --dir runs/largest_virtual/10_7 --loop_num 1
#python train.py --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 7 --dir runs/largest_virtual/10_7 --loop_num 2
