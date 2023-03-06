#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
# server='localhost'
server='16.162.205.1'
# clients='18.163.112.214 18.162.149.191 16.163.141.88 16.162.45.130 18.166.225.51 18.162.111.137 16.163.146.64 18.162.111.65 18.163.182.179 18.166.178.80 18.166.30.86 43.198.86.162 18.163.182.123 18.162.154.99 18.163.124.20 16.163.143.211 18.163.114.212 16.162.45.177 18.167.20.19 18.166.62.148 18.163.112.214 18.162.149.191 16.163.141.88 16.162.45.130 18.166.225.51 18.162.111.137 16.163.146.64 18.162.111.65 18.163.182.179 18.166.178.80 18.166.30.86 43.198.86.162 18.163.182.123 18.162.154.99 18.163.124.20 16.163.143.211 18.163.114.212 16.162.45.177 18.167.20.19 18.166.62.148 18.163.112.214 18.162.149.191 16.163.141.88 16.162.45.130 18.166.225.51 18.162.111.137 16.163.146.64 18.162.111.65 18.163.182.179 18.166.178.80 18.166.30.86 43.198.86.162 18.163.182.123 18.162.154.99 18.163.124.20 16.163.143.211 18.163.114.212 16.162.45.177 18.167.20.19 18.166.62.148 18.163.112.214 18.162.149.191 16.163.141.88 16.162.45.130 18.166.225.51 18.162.111.137 16.163.146.64 18.162.111.65 18.163.182.179 18.166.178.80 18.166.30.86 43.198.86.162 18.163.182.123 18.162.154.99 18.163.124.20 16.163.143.211 18.163.114.212 16.162.45.177 18.167.20.19 18.166.62.148'
clients='18.166.55.130 18.167.85.7 18.163.111.195 18.166.63.24 16.162.55.143 43.198.71.196 18.167.68.174 18.163.35.5 18.162.214.72 16.163.57.106 18.163.111.48 18.167.103.121 18.166.55.75 43.198.71.241 16.163.52.171 18.162.134.98 16.163.42.208 18.166.65.224 18.166.74.175 16.162.86.7'
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
# model="MobileNetV2_Cifar10"
model="LSTM_KWS"
# model="CNN_Cifar10"

### dateset
dataset="KWS"
# dataset="Cifar10"

### interlayer type
interlayer_type="Default"
# interlayer_type="CAS"

### sync setting
batchsize=64
init_period=64
dirichlet_alpha=0.1

### --- launch server process
server_job=server_process_expotential.py
# server_job=server_process_linear.py
command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/$server_job \
	--port=$((20000+trial_no)) --world_size=$world_size  --init_period=$init_period \
	--remarks=$remarks --trial_no=$trial_no --interlayer_type=$interlayer_type \
    --model=$model --dataset=$dataset"
echo "launch server:"
echo $command
echo
# nohup ssh $server $command >> $log_dir/client_process_1.log 2>&1 &
nohup ssh $server $command > $log_dir/server_process.log 2>&1 &
sleep 5

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




