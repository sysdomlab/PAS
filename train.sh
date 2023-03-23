#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
# server='localhost'
server='18.166.161.152'
clients="43.198.103.129 18.163.81.207 43.198.107.75 18.167.42.217 18.162.45.215 16.162.32.198 18.162.39.209 16.163.31.198 18.166.208.55 16.163.101.50 16.162.89.148 18.163.198.107 16.162.161.65 18.162.123.69 18.163.194.148 18.163.188.130 18.162.42.148 18.162.156.244 43.198.109.43 18.163.238.111 \
         16.162.253.141 43.198.95.79 18.167.172.52 18.166.61.207 16.163.102.174 43.198.108.55 18.162.41.216 43.198.97.239 18.162.200.133 18.162.194.201 43.198.111.217 18.162.45.107 43.198.110.198 18.162.190.164 18.162.42.134 16.163.99.101 16.163.29.66 18.163.33.182 18.167.173.43 18.166.212.64 \
         18.163.6.39 18.167.69.202 18.166.209.60 18.163.46.163 18.162.45.61 18.166.61.6 43.198.99.139 16.163.156.70 18.163.40.76 16.162.255.128 16.163.99.147 18.167.126.73 43.198.98.170 16.162.3.119 18.166.208.89 16.162.19.213 18.163.194.137 18.163.125.181 43.198.102.195 18.162.187.74 \
         16.162.254.136 18.167.59.244 16.162.160.61 18.163.214.147 16.163.157.193 16.162.93.203 18.162.188.103 18.166.211.2 18.167.99.142 16.162.90.248 18.163.40.162 18.167.54.183 18.162.133.156 43.198.106.14 18.162.191.42 43.198.90.156 16.162.161.99 18.162.39.92 43.198.110.73 18.166.223.89 \
         43.198.90.46 18.163.24.217 18.162.244.217 18.163.194.107 18.163.75.243 16.163.101.115 18.162.43.252 43.198.106.119 18.166.27.228 18.162.232.185 18.166.211.113 18.166.61.30 16.163.105.115 18.163.198.177 16.163.30.114 16.163.127.128 18.162.244.206 18.166.61.151 18.167.167.12 18.167.41.189"
# clients="18.163.105.212 18.162.62.213 43.198.71.162 18.163.111.207 18.163.101.42 18.163.50.216 18.163.121.50 18.166.63.18 43.198.72.156 18.162.148.230 18.162.55.17 18.163.117.50 16.163.108.47 18.163.129.245 18.162.155.115 16.162.91.252 43.198.21.64 18.166.78.138 43.198.75.185 43.198.73.38"
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
# model="ResNet20_Cifar10"
# model="ResNet18_Cifar10"
# model="MobileNetV2_Cifar10"
# model="LSTM_KWS"
# model="CNN_Cifar10"
# model="VGG11_Cifar10"
# model="CNN_Mnist"
model="CNN_EMnist"
# model="CNN_FashionMnist"

### dateset
# dataset="KWS"
# dataset="Cifar10"
dataset="EMnist"
# dataset="FashionMnist"

### interlayer type
# interlayer_type="Default"
interlayer_type="CAS"

### tune_method
method="expotential"
# method="linear"

work_dir=$home'/'$method
echo $work_dir
echo $home

### sync setting
batchsize=200
init_period=200
dirichlet_alpha=0.01

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




