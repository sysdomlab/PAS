#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
# master='18.167.0.57'
server='localhost'
# workers='18.167.0.57 18.163.188.30 18.162.44.220 43.198.95.97 16.163.156.117 18.162.123.143 18.162.44.79 18.167.134.169 43.198.96.81 18.163.24.92 16.163.1.98 18.162.120.27 18.167.51.97 43.198.95.178 18.162.43.201 18.163.194.43 18.162.142.216 16.162.88.146 43.198.90.67 18.167.165.118'


clients='localhost localhost localhost localhost'

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



