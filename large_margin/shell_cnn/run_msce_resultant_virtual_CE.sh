cd ../

python train.py --network cnn --dataset mnist --approach msce_resultant_virtual --loss_type CE --learning_rate 0.01  --batch_size 128 --epochs 100 --dir runs/msce_resultant_virtual_CE/mnist --loop_num 0
python train.py --network cnn --dataset mnist --approach msce_resultant_virtual --loss_type CE --learning_rate 0.01  --batch_size 128 --epochs 100 --dir runs/msce_resultant_virtual_CE/mnist --loop_num 1
python train.py --network cnn --dataset mnist --approach msce_resultant_virtual --loss_type CE --learning_rate 0.01  --batch_size 128 --epochs 100 --dir runs/msce_resultant_virtual_CE/mnist --loop_num 2
