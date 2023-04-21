#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
# server='localhost'
server='18.166.161.152'
clients="43.198.101.204 16.162.252.241 18.162.133.190 18.162.156.192 43.198.107.219 18.163.198.14 16.163.105.146 16.162.21.106 43.198.88.158 18.162.123.234 18.166.61.2 43.198.101.156 18.163.46.141 18.167.166.184 18.163.210.25 18.163.198.94 43.198.102.7 18.167.59.170 16.163.94.64 18.163.2.50 \
         43.198.90.182 18.162.191.3 18.162.41.90 16.163.98.205 18.163.40.184 18.163.214.108 43.198.107.75 16.162.161.190 18.162.194.13 18.163.2.48 43.198.101.241 18.163.238.15 18.162.45.54 18.162.232.193 18.167.166.91 16.163.104.74 18.163.214.146 43.198.104.83 16.163.94.14 43.198.101.233"
# clients="16.162.253.208 18.163.208.126 43.198.97.21 18.163.125.112 43.198.89.192 43.198.96.62 16.162.254.68 16.163.94.120 16.162.3.192 18.163.24.72 18.162.191.238 18.167.69.35 18.162.42.109 18.167.173.216 43.198.99.104 43.198.90.119 43.198.107.125 18.162.228.100 16.163.30.102 18.162.45.2 \
#          18.163.49.102 18.163.197.50 18.167.54.109 43.198.102.196 16.162.46.252 18.163.81.252 43.198.97.139 18.162.200.244 16.162.46.54 18.166.15.97 18.166.28.155 18.163.125.22 18.167.173.233 43.198.110.112 43.198.89.220 18.162.97.109 18.163.6.253 18.162.200.1 18.166.211.79 43.198.100.231"
# clients='localhost localhost'
# clients='18.166.73.214 18.162.42.252 18.162.228.74 18.167.99.138'

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
# interlayer_type="Default"
interlayer_type="CAS"
# interlayer_type="GIFT"

### tune_method
# method="expotential"
method="linear"
# method="expotential_without_limit"
# method="layerwise"

work_dir=$home'/'$method
cp -r $work_dir $log_dir/code_snapshot
echo $work_dir
echo $home

### sync setting
batchsize=100
init_period=200
dirichlet_alpha=0.05

### --- launch server process
command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $work_dir/server_test_process.py \
	--port=$((10000+trial_no)) --remarks=$remarks --trial_no=$trial_no \
    --model=$model --dataset=$dataset"
echo $command
nohup ssh $server $command >> $log_dir/server_process.log 2>&1 &
sleep 5

command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $work_dir/server_process.py \
	--port=$((20000+trial_no)) --world_size=$world_size  --init_period=$init_period \
	--remarks=$remarks --trial_no=$trial_no --interlayer_type=$interlayer_type \
    --test_process_port=$((10000+trial_no))"
echo "launch server:"
echo $command
echo
# nohup ssh $server $command >> $log_dir/client_process_1.log 2>&1 &
nohup ssh $server $command >> $log_dir/server_process.log 2>&1 &
sleep 5

### --- launch sync process
if [ "$interlayer_type" = "CAS" ]; then
    num=0
    for i in $clients; do
        command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $work_dir/sync_process.py \
            --server_ip=$server --server_port=$((20000+trial_no)) --rank=$num \
            --port=$((30000+ trial_no + 10*num)) --remarks=$remarks --trial_no=$trial_no"
        echo $command
        nohup ssh $i $command >> $log_dir/client_process_$num.log 2>&1 &
        num=$((num+1))
        sleep 0.2
    done
    sleep 1
    ### --- launch client process
    num=0
    for i in $clients; do
        command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $work_dir/worker_process.py \
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
else 
    ### --- launch client process
    num=0
    for i in $clients; do
        command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $work_dir/worker_process.py \
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
fi




