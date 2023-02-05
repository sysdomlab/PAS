#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
master='18.162.155.234'
# master='localhost'
workers='18.162.155.234 18.162.45.16 18.163.116.86 18.166.176.205 16.162.47.226 18.162.154.167 43.198.13.129 18.163.56.89 18.162.230.55 16.163.140.148 43.198.15.247 18.166.29.93 16.162.109.34 16.163.129.15 18.162.110.27 18.163.112.138 18.162.47.90 18.162.167.16 43.198.81.107 18.163.7.117'

# workers='localhost localhost'

echo 'master(coordinator): '$master
echo 'worker_hosts: '$workers
world_size=0
for i in $workers; do
    world_size=$((world_size+1))
done

### create log dir and code snapshot
read -p "Enter exp-setup notes: " remarks
read -p "Specify allocated GPU-ID (world_size: $world_size): " cuda
trial_no=$(ls $home/Logs/ | wc -l)
log_dir=$home/Logs/${trial_no}_$remarks

mkdir -p $log_dir/code_snapshot
cp $home/train.sh $log_dir/code_snapshot
cp $home/*.py $log_dir/code_snapshot
cp -r $home/models $log_dir/code_snapshot
rm -rf Latest_Log && ln -s $log_dir Latest_Log
echo 'logs in '$log_dir

### launch process
num=0
job=worker_process.py
model="LSTM_KWS"
dataset="KWS"
interlayer_type="ParamGIFT"
batchsize=20
init_frequency=32
for i in $workers; do
    command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/$job --master_address=tcp://${master}:$((20000+trial_no)) --rank=$num --world_size=$world_size --remarks=$remarks --trial_no=$trial_no --model=$model --dataset=$dataset --interlayer_type=$interlayer_type --batchsize=$batchsize --init_frequency=$init_frequency"
    echo $command
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done




