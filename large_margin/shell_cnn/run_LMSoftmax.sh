cd ../

python train.py --network cnn --dataset mnist --approach sce --loss_type LMSoftmax --learning_rate 0.01  --batch_size 128 --epochs 100 --dir ./runs_LMSoftmax/mnist --loop_num 0
python train.py --network cnn --dataset mnist --approach sce --loss_type LMSoftmax --learning_rate 0.01  --batch_size 128 --epochs 100 --dir ./runs_LMSoftmax/mnist --loop_num 1
python train.py --network cnn --dataset mnist --approach sce --loss_type LMSoftmax --learning_rate 0.01  --batch_size 128 --epochs 100 --dir ./runs_LMSoftmax/mnist --loop_num 2
