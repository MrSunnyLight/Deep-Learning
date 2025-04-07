cd ../

python train.py --network cnn --dataset mnist --approach virtual_learning_strategy --loss_type Focal --learning_rate 0.01 --batch_size 128 --epochs 100 --dir runs/*virtual_learning_strategy_useat100 --loop_num 0
python train.py --network cnn --dataset mnist --approach virtual_learning_strategy --loss_type Focal --learning_rate 0.01 --batch_size 128 --epochs 100 --dir runs/*virtual_learning_strategy_useat100 --loop_num 1
python train.py --network cnn --dataset mnist --approach virtual_learning_strategy --loss_type Focal --learning_rate 0.01 --batch_size 128 --epochs 100 --dir runs/*virtual_learning_strategy_useat100 --loop_num 2
