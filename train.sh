#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
# server='localhost'
server='16.162.205.1'
clients='18.162.144.155 16.162.188.0 43.198.79.184 18.166.69.254 18.162.149.82 43.198.11.120 18.162.210.45 43.198.14.96 16.163.145.139 16.162.137.238 43.198.12.20 18.162.207.246 16.162.122.205 18.166.75.17 43.198.84.151 18.162.149.71 18.162.141.170 16.162.118.114 43.198.83.15 18.163.28.157'
# clients='localhost localhost'

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

### sync setting
batchsize=64
init_period=64
dirichlet_alpha=1

### --- launch server process
command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/server_process.py \
	--port=$((20000+trial_no)) --world_size=$world_size  --init_period=$init_period \
	--remarks=$remarks --trial_no=$trial_no --interlayer_type=$interlayer_type \
    --model=$model --dataset=$dataset"
echo "launch server:"
echo $command
echo
# nohup ssh $server $command >> $log_dir/client_process_1.log 2>&1 &
nohup ssh $server $command >> $log_dir/server_process.log 2>&1 &
sleep 1

### --- launch sync process
if [ "$interlayer_type" = "CAS" ]; then
    num=0
    for i in $clients; do
        command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/sync_process.py \
            --server_ip=$server --server_port=$((20000+trial_no)) --rank=$num \
            --port=$((30000+ trial_no + 10*num)) --remarks=$remarks --trial_no=$trial_no"
        echo $command
        nohup ssh $i $command >> $log_dir/client_process_$num.log 2>&1 &
        num=$((num+1))
    done
    sleep 1
    ### --- launch client process
    num=0
    for i in $clients; do
        command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/worker_process.py \
            --sync_process_port=$((30000 + trial_no + 10*num))\
            --server_ip=$server --server_port=$((20000+trial_no)) --rank=$num \
            --world_size=$world_size --remarks=$remarks --trial_no=$trial_no \
            --model=$model --dataset=$dataset --interlayer_type=$interlayer_type \
            --batchsize=$batchsize --init_period=$init_period \
            --dirichlet_alpha=$dirichlet_alpha"
        echo $command
        nohup ssh $i $command >> $log_dir/client_process_$num.log 2>&1 &
        num=$((num+1))
    done
else 
    ### --- launch client process
    num=0
    for i in $clients; do
        command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/worker_process.py \
            --sync_process_port=$((30000 + trial_no + 10*num))\
            --server_ip=$server --server_port=$((20000+trial_no)) --rank=$num \
            --world_size=$world_size --remarks=$remarks --trial_no=$trial_no \
            --model=$model --dataset=$dataset --interlayer_type=$interlayer_type \
            --batchsize=$batchsize --init_period=$init_period \
            --dirichlet_alpha=$dirichlet_alpha"
        echo $command
        nohup ssh $i $command >> $log_dir/client_process_$num.log 2>&1 &
        num=$((num+1))
    done
fi




