#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
# master='18.167.0.57'
server='18.167.0.57'
clients='18.162.55.161 18.166.67.172 18.163.77.191 16.163.40.18 18.163.50.185 18.162.116.86 18.163.73.147 18.162.114.6 43.198.24.254 16.163.103.16 18.163.129.160 43.198.74.74 18.163.8.137 18.166.78.151 18.166.54.95 18.167.109.223 18.166.51.117 18.163.121.81 43.198.19.64 18.163.123.206'
# clients='localhost localhost localhost localhost'

echo 'Server_host: '$server
echo 'Client_hosts: '$clients
world_size=0
for i in $clients; do
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

### model
# model="ResNet18_Cifar10"
model="LSTM_KWS"
# model="CNN_Cifar10"

### dateset
dataset="KWS"
# dataset="Cifar10"

### interlayer type
# interlayer_type="Default"
interlayer_type="CAS"

### --- launch server process
command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/server_process.py \
	--port=$((20000+trial_no)) --world_size=$world_size\
	--remarks=$remarks --trial_no=$trial_no --interlayer_type=$interlayer_type"
echo "launch server:"
echo $command
echo
nohup ssh $server $command > $log_dir/server.log 2>&1 &
sleep 1

### --- launch client process
num=0
job=worker_process.py
batchsize=64
init_period=64
dirichlet_alpha=0.5
for i in $clients; do
    command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/$job \
        --server_ip=$server --server_port=$((20000+trial_no)) --rank=$num \
        --world_size=$world_size --remarks=$remarks --trial_no=$trial_no \
        --model=$model --dataset=$dataset --interlayer_type=$interlayer_type \
        --batchsize=$batchsize --init_period=$init_period \
        --dirichlet_alpha=$dirichlet_alpha"
    echo $command
    nohup ssh $i $command > $log_dir/client_$num.log 2>&1 &
    num=$((num+1))
done



