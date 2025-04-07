cd ../

python train.py --network cnn --dataset mnist --approach sce_rsm --learning_rate 0.01 --epochs 10 --dir runs/sce_rsm/mnist --loop_num 0
# python train.py --network cnn --dataset mnist --approach sce_rsm --learning_rate 0.01 --epochs 100 --dir runs/sce_rsm/mnist --loop_num 1
# python train.py --network cnn --dataset mnist --approach sce_rsm --learning_rate 0.01 --epochs 100 --dir runs/sce_rsm/mnist --loop_num 2
