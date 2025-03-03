#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
server='192.168.0.9'
# server='localhost'
# clients='192.168.0.239 192.168.0.240 192.168.0.242 192.168.0.241'
# clients='localhost localhost'
clients="192.168.0.12 192.168.0.11 192.168.0.18 192.168.0.15 192.168.0.10 192.168.0.17 192.168.0.13 192.168.0.24 192.168.0.38 192.168.0.16 192.168.0.34 192.168.0.25 192.168.0.29 192.168.0.42 192.168.0.14 192.168.0.20 192.168.0.44 192.168.0.22 192.168.0.23 192.168.0.43 192.168.0.19 192.168.0.30 192.168.0.28 192.168.0.21 192.168.0.40 192.168.0.46 192.168.0.33 192.168.0.36 192.168.0.31 192.168.0.45 192.168.0.27 192.168.0.26 192.168.0.35 192.168.0.32 192.168.0.39 192.168.0.37 192.168.0.47 192.168.0.49 192.168.0.48 192.168.0.41"

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
# model="ResNet20_Cifar10"
# model="ResNet18_Cifar10"
# model="MobileNetV2_Cifar10"
model="LSTM_KWS"
# model="CNN_Cifar10"
# model="VGG11_Cifar10"
# model="CNN_Mnist"
# model="CNN_EMnist"
# model="CNN_FashionMnist"

### dateset
dataset="KWS"
# dataset="Cifar10"
# dataset="EMnist"
# dataset="FashionMnist"

### interlayer type
# interlayer_type="FedAvg"
# interlayer_type="FedProx"
# interlayer_type="AdaComm"
# interlayer_type="FedLAMA"
# interlayer_type="CAS"
interlayer_type="CAS-APF"
# interlayer_type="GIFT"


### tune_method
method="expotential"
# method="linear"
# method="expotential_without_limit"
# method="layerwise"

work_dir=$home'/'$method
cp -r $work_dir $log_dir/code_snapshot
echo $work_dir
echo $home

### sync setting
batchsize=100
init_period=200
dirichlet_alpha=0.5
margin=10

### --- launch server process
command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $work_dir/server/server_test_process.py \
	--port=$((10000+trial_no)) --remarks=$remarks --trial_no=$trial_no \
    --model=$model --dataset=$dataset"
echo $command
nohup ssh $server $command >> $log_dir/server_process.log 2>&1 &
sleep 5

command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $work_dir/server/server_process.py \
	--port=$((20000+trial_no)) --world_size=$world_size  --init_period=$init_period \
	--remarks=$remarks --trial_no=$trial_no --interlayer_type=$interlayer_type \
    --test_process_port=$((10000+trial_no))"
echo "launch server:"
echo $command
echo
# nohup ssh $server $command >> $log_dir/client_process_1.log 2>&1 &
nohup ssh $server $command >> $log_dir/server_process.log 2>&1 &
sleep 5


if [[ $interlayer_type =~ ^CAS ]]; then
    num=0
    ### --- launch sync process
    for i in $clients; do
        command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $work_dir/client/cas_sync_process.py \
            --server_ip=$server --server_port=$((20000+trial_no)) --rank=$num \
            --port=$((30000+ trial_no + 10*num)) --remarks=$remarks --trial_no=$trial_no"
        echo $command
        nohup ssh $i $command >> $log_dir/client_process_$num.log 2>&1 &
        num=$((num+1))
        sleep 0.2
    done
    sleep 1
fi


### --- launch client process
num=0
for i in $clients; do
    command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $work_dir/client/worker_process.py \
        --sync_process_port=$((30000 + trial_no + 10*num))\
        --server_ip=$server --server_port=$((20000+trial_no)) --rank=$num \
        --world_size=$world_size --remarks=$remarks --trial_no=$trial_no \
        --model=$model --dataset=$dataset --interlayer_type=$interlayer_type \
        --batchsize=$batchsize --init_period=$init_period \
        --dirichlet_alpha=$dirichlet_alpha"
    echo $command
    nohup ssh $i $command >> $log_dir/client_process_$num.log 2>&1 &
    num=$((num+1))
    sleep 0.2
done




