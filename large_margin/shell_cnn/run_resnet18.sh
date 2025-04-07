cd ../
# sce
python train.py --network resnet18 --dataset cifar10 --approach sce --batch_size 128 --epochs 200 --dir runs/r18/sce --loop_num 0
python train.py --network resnet18 --dataset cifar10 --approach sce --batch_size 128 --epochs 200 --dir runs/r18/sce --loop_num 1
python train.py --network resnet18 --dataset cifar10 --approach sce --batch_size 128 --epochs 200 --dir runs/r18/sce --loop_num 2

# sce + focal
python train.py --network resnet18 --dataset cifar10 --approach sce --loss_type Focal --batch_size 128 --epochs 200 --dir runs/r18/focal --loop_num 0
python train.py --network resnet18 --dataset cifar10 --approach sce --loss_type Focal --batch_size 128 --epochs 200 --dir runs/r18/focal --loop_num 1
python train.py --network resnet18 --dataset cifar10 --approach sce --loss_type Focal --batch_size 128 --epochs 200 --dir runs/r18/focal --loop_num 2

# msce
python train.py --network resnet18 --dataset cifar10 --approach msce --batch_size 128 --epochs 200 --dir runs/r18/msce --loop_num 0
python train.py --network resnet18 --dataset cifar10 --approach msce --batch_size 128 --epochs 200 --dir runs/r18/msce --loop_num 1
python train.py --network resnet18 --dataset cifar10 --approach msce --batch_size 128 --epochs 200 --dir runs/r18/msce --loop_num 2

# virtual softmax
python train.py --network resnet18 --dataset cifar10 --approach virtual_softmax --batch_size 128 --epochs 200 --dir runs/r18/virtual_softmax --loop_num 0
python train.py --network resnet18 --dataset cifar10 --approach virtual_softmax --batch_size 128 --epochs 200 --dir runs/r18/virtual_softmax --loop_num 1
python train.py --network resnet18 --dataset cifar10 --approach virtual_softmax --batch_size 128 --epochs 200 --dir runs/r18/virtual_softmax --loop_num 2

# virtual focal
python train.py --network resnet18 --dataset cifar10 --approach virtual_focal --loss_type Focal --batch_size 128 --epochs 200 --dir runs/r18/virtual_focal --loop_num 0
python train.py --network resnet18 --dataset cifar10 --approach virtual_focal --loss_type Focal --batch_size 128 --epochs 200 --dir runs/r18/virtual_focal --loop_num 1
python train.py --network resnet18 --dataset cifar10 --approach virtual_focal --loss_type Focal --batch_size 128 --epochs 200 --dir runs/r18/virtual_focal --loop_num 2

# resultant_virtual Focal
python train.py --network resnet18 --dataset cifar10 --approach resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/r18/resultant_virtual_focal --loop_num 0
python train.py --network resnet18 --dataset cifar10 --approach resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/r18/resultant_virtual_focal --loop_num 1
python train.py --network resnet18 --dataset cifar10 --approach resultant_virtual --loss_type Focal --batch_size 128 --epochs 200 --dir runs/r18/resultant_virtual_focal --loop_num 2

# virtual_learning_strategy_focal
python train.py --network resnet18 --dataset cifar10 --approach virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/r18/virtual_learning_strategy --loop_num 0
python train.py --network resnet18 --dataset cifar10 --approach virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/r18/virtual_learning_strategy --loop_num 1
python train.py --network resnet18 --dataset cifar10 --approach virtual_learning_strategy --loss_type Focal --batch_size 128 --epochs 200 --dir runs/r18/virtual_learning_strategy --loop_num 2