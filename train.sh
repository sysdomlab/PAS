#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
master='18.166.51.62'
master='localhost'
workers='18.166.51.62 18.166.51.62 18.163.55.176 18.163.55.176 16.163.55.57 16.163.55.57 43.198.31.5 43.198.31.5 18.166.76.65 18.166.76.65 18.166.54.10 18.166.54.10 18.162.55.73 18.162.55.73 18.162.155.75 18.162.155.75 16.163.50.241 16.163.50.241 16.162.25.13 16.162.25.13'
workers='18.166.51.62 18.166.51.62 18.166.51.62 18.166.51.62'
workers='localhost localhost'

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
batchsize=100
init_frequency=64
for i in $workers; do
    command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/$job --master_address=tcp://${master}:$((20000+trial_no)) --rank=$num --world_size=$world_size --remarks=$remarks --trial_no=$trial_no --model=$model --dataset=$dataset --interlayer_type=$interlayer_type --batchsize=$batchsize --init_frequency=$init_frequency"
    echo $command
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done




