cd ../

python train.py --network cnn --dataset mnist --approach resultant_virtual --loss_type Focal --learning_rate 0.01 --batch_size 128 --epochs 100 --dir runs/mnist_resultant_virtual_focal --loop_num 0
python train.py --network cnn --dataset mnist --approach resultant_virtual --loss_type Focal --learning_rate 0.01 --batch_size 128 --epochs 100 --dir runs/mnist_resultant_virtual_focal --loop_num 1
python train.py --network cnn --dataset mnist --approach resultant_virtual --loss_type Focal --learning_rate 0.01 --batch_size 128 --epochs 100 --dir runs/mnist_resultant_virtual_focal --loop_num 2
