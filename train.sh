#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
# server='localhost'
server='18.166.161.152'
clients="18.162.39.66 43.198.94.65 18.167.42.192 16.162.255.142 18.166.211.63 16.163.156.254 18.162.190.56 18.166.211.216 16.163.29.95 43.198.102.185 18.163.79.227 18.163.75.36 18.163.200.5 18.162.40.116 43.198.89.211 16.163.123.4 16.163.156.84 43.198.102.205 18.162.42.148 16.162.4.100 \
         16.163.105.186 18.163.24.159 18.163.49.236 18.163.215.37 18.163.188.200 18.163.215.138 43.198.89.131 18.162.228.59 18.163.128.54 16.163.29.136 16.162.93.133 18.166.27.135 18.167.99.148 18.162.153.171 18.163.208.132 18.162.187.151 18.163.40.230 18.162.145.177 18.163.33.202 18.162.244.10 \
         18.162.142.115 18.163.46.81 18.163.122.184 16.163.31.88 16.163.101.93 18.162.39.108 18.166.27.55 43.198.107.163 18.162.46.183 18.163.196.197 43.198.94.68 18.166.212.187 18.166.213.45 18.167.173.205 43.198.100.144 18.166.211.113 18.163.40.66 18.162.145.88 43.198.101.152 16.162.46.146 \
         16.162.254.181 18.163.208.139 18.163.6.43 16.163.157.60 18.163.206.108 16.163.95.95 43.198.88.209 18.166.208.126 16.163.99.37 18.162.142.60 18.162.228.121 18.167.59.75 16.162.90.228 18.163.206.120 18.163.33.246 16.163.101.247 18.162.97.169 18.166.27.225 18.163.81.62 18.167.51.216 \
         18.167.172.148 43.198.89.165 18.162.200.49 16.162.23.162 18.167.126.5 16.163.127.198 18.167.99.77 18.162.120.105 18.166.28.96 18.163.46.45 18.163.196.64 18.162.200.83 16.162.4.7 16.162.93.201 16.162.93.167 18.163.24.239 18.162.40.208 18.163.128.151 16.162.88.64 18.167.166.214"
# clients="16.162.253.208 18.163.208.126 43.198.97.21 18.163.125.112 43.198.89.192 43.198.96.62 16.162.254.68 16.163.94.120 16.162.3.192 18.163.24.72 18.162.191.238 18.167.69.35 18.162.42.109 18.167.173.216 43.198.99.104 43.198.90.119 43.198.107.125 18.162.228.100 16.163.30.102 18.162.45.2 \
#          18.163.49.102 18.163.197.50 18.167.54.109 43.198.102.196 16.162.46.252 18.163.81.252 43.198.97.139 18.162.200.244 16.162.46.54 18.166.15.97 18.166.28.155 18.163.125.22 18.167.173.233 43.198.110.112 43.198.89.220 18.162.97.109 18.163.6.253 18.162.200.1 18.166.211.79 43.198.100.231"
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

### tune_method
# method="expotential"
# method="linear"
method="expotential_without_limit"

work_dir=$home'/'$method
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




