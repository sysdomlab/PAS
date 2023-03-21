#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
server='localhost'
# server='18.166.164.215'
# clients="18.162.155.232 16.163.103.248 43.198.78.113 18.162.214.161 43.198.74.228 18.163.187.84 18.162.193.248 16.163.43.167 16.162.48.62 18.163.189.132 18.162.57.180 18.162.55.117 18.166.52.62 18.163.8.192 18.166.58.81 43.198.78.98 18.166.54.38 18.162.151.239 43.198.22.93 18.167.109.96 \
#          43.198.19.36 18.163.35.21 18.162.155.190 18.166.55.86 43.198.21.174 18.163.5.205 18.163.55.65 18.163.105.40 18.166.76.92 18.166.53.165 16.163.55.207 18.163.41.14 18.162.56.96 16.163.41.238 43.198.25.146 16.163.55.233 18.163.117.185 18.163.110.107 18.162.114.177 18.162.126.68 \
#          18.167.120.50 18.166.55.234 18.166.67.159 18.163.183.186 43.198.78.30 18.162.60.4 16.162.91.86 18.163.105.96 18.167.90.213 43.198.78.6 16.163.103.55 18.163.190.72 18.162.155.191 18.163.118.175 16.163.40.103 16.162.86.216 18.162.155.215 18.163.41.207 16.162.44.95 43.198.74.152 \
#          18.163.105.8 18.167.96.173 18.166.78.79 18.167.84.165 16.163.42.118 16.162.25.91 18.167.116.208 43.198.78.191 18.163.183.5 18.162.62.27 18.166.51.244 43.198.74.8 16.162.44.188 18.163.102.193 18.162.61.76 16.163.58.198 18.167.121.244 18.166.67.10 16.163.109.44 18.167.85.203 \
#          18.162.55.91 18.163.62.16 18.162.115.3 43.198.24.180 18.163.187.189 18.166.54.139 18.163.181.185 18.163.185.86 18.166.51.105 18.167.84.37 18.162.134.8 18.162.146.57 16.163.108.252 18.163.62.157 18.167.103.75 18.163.187.179 18.162.143.44 16.162.25.163 18.162.126.57 18.166.57.152"
# clients="18.163.105.212 18.162.62.213 43.198.71.162 18.163.111.207 18.163.101.42 18.163.50.216 18.163.121.50 18.166.63.18 43.198.72.156 18.162.148.230 18.162.55.17 18.163.117.50 16.163.108.47 18.163.129.245 18.162.155.115 16.162.91.252 43.198.21.64 18.166.78.138 43.198.75.185 43.198.73.38"
clients='localhost localhost'

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
# model="CNN_EMnist"
model="CNN_FashionMnist"

### dateset
# dataset="KWS"
# dataset="Cifar10"
# dataset="EMnist"
dataset="FashionMnist"

### interlayer type
interlayer_type="Default"
# interlayer_type="CAS"

method="expotential"

work_dir=$home'/'$method
echo $work_dir
echo $home

### sync setting
batchsize=100
init_period=200
dirichlet_alpha=0.05

### --- launch server process
command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $work_dir/server_test_process.py \
	--port=$((30000+trial_no)) --remarks=$remarks --trial_no=$trial_no \
    --model=$model --dataset=$dataset"
echo $command
nohup ssh $server $command >> $log_dir/server_process.log 2>&1 &
sleep 1

command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $work_dir/server_process.py \
	--port=$((20000+trial_no)) --world_size=$world_size  --init_period=$init_period \
	--remarks=$remarks --trial_no=$trial_no --interlayer_type=$interlayer_type \
    --test_process_port=$((30000+trial_no))"
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




