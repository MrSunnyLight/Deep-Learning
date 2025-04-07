# cd ../

venv/Scripts/python.exe train.py --device cuda:0 --network resnet34 --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 8 --scale 0.1  --dir runs/largest_virtual/06_16 --loop_num 0
venv/Scripts/python.exe train.py --device cuda:0 --network resnet34 --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 8 --scale 0.3  --dir runs/largest_virtual/06_16 --loop_num 1
venv/Scripts/python.exe train.py --device cuda:0 --network resnet34 --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 8 --scale 0.5  --dir runs/largest_virtual/06_16 --loop_num 2
venv/Scripts/python.exe train.py --device cuda:0 --network resnet34 --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 8 --scale 0.6  --dir runs/largest_virtual/06_16 --loop_num 3
venv/Scripts/python.exe train.py --device cuda:0 --network resnet34 --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 8 --scale 0.7  --dir runs/largest_virtual/06_16 --loop_num 4
venv/Scripts/python.exe train.py --device cuda:0 --network resnet34 --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 8 --scale 0.8  --dir runs/largest_virtual/06_16 --loop_num 5
venv/Scripts/python.exe train.py --device cuda:0 --network resnet34 --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 8 --scale 0.9  --dir runs/largest_virtual/06_16 --loop_num 6
venv/Scripts/python.exe train.py --device cuda:0 --network resnet34 --dataset cifar10 --approach largest_virtual --batch_size 128 --epochs 200 --select 8 --scale 1.0  --dir runs/largest_virtual/06_16 --loop_num 7
/usr/bin/shutdown


