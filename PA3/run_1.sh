# Note: all paths referenced here are relative to the Docker container.
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
source /tools/config.sh
source activate py27
cd /storage/home/karthikt/DL/PA3
# Learning rate experiments
nohup python -u scripts/main.py --arch models/1.json --lr 0.01 --batch_size 20 --init 1 --model_name 1.1 &
nohup python -u scripts/main.py --arch models/1.json --lr 0.001 --batch_size 20 --init 1 --model_name 1.2 &
nohup python -u scripts/main.py --arch models/1.json --lr 0.0001 --batch_size 20 --init 1 --model_name 1.3 &

# Initializer experiment
nohup python -u scripts/main.py --arch models/1.json --lr 0.001 --batch_size 50 --init 1 --model_name 1.7 &

sleep 86400
