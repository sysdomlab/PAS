#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
# master='16.163.55.246'
master='localhost'
# workers='16.163.55.246 18.163.84.174 18.166.71.35 16.162.106.7 18.162.48.187 18.166.75.148 43.198.16.173 43.198.13.209 18.162.152.136 16.163.106.124 16.162.87.196 43.198.86.121 43.198.11.127 16.162.105.36 16.162.22.184 18.162.168.223 18.166.31.49 18.163.195.198 16.162.45.244 18.162.230.3'

workers='localhost localhost localhost localhost'

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
# model="ResNet18_Cifar10"
model="LSTM_KWS"
dataset="KWS"
# dataset="Cifar10"
# interlayer_type="Default"
interlayer_type="ParamGIFT"
batchsize=20
init_frequency=1
for i in $workers; do
    command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/$job --master_address=tcp://${master}:$((20000+trial_no)) --rank=$num --world_size=$world_size --remarks=$remarks --trial_no=$trial_no --model=$model --dataset=$dataset --interlayer_type=$interlayer_type --batchsize=$batchsize --init_frequency=$init_frequency"
    echo $command
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done




