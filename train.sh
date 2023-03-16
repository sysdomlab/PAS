#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
# server='localhost'
server='16.162.205.1'
clients="18.163.105.212 18.162.62.213 43.198.71.162 18.163.111.207 18.163.101.42 18.163.50.216 18.163.121.50 18.166.63.18 43.198.72.156 18.162.148.230 18.162.55.17 18.163.117.50 16.163.108.47 18.163.129.245 18.162.155.115 16.162.91.252 43.198.21.64 18.166.78.138 43.198.75.185 43.198.73.38 \
         16.163.109.25 43.198.72.172 18.163.187.104 43.198.24.101 18.162.56.255 16.163.103.147 43.198.73.124 18.163.41.135 18.167.94.114 18.163.102.64 18.167.94.2 16.162.25.141 16.163.128.229 18.163.187.0 18.162.61.238 18.163.102.192 18.163.189.104 18.166.57.138 18.162.115.237 16.162.91.241 \
         16.162.44.181 18.163.110.173 18.163.102.159 18.166.74.54 18.163.102.98 16.162.48.123 18.163.111.239 18.166.72.112 18.162.209.71 18.166.65.141 16.163.128.154 18.166.53.63 18.162.225.189 43.198.76.200 16.163.52.69 43.198.76.175 18.163.110.215 18.166.51.143 18.163.187.139 18.163.62.23 \
         18.163.77.43 43.198.24.5 18.162.114.229 18.163.183.191 18.163.129.50 43.198.75.23 43.198.23.190 43.198.73.226 18.167.72.25 43.198.75.39 18.163.101.18 43.198.71.3 18.167.90.37 18.167.109.212 18.167.84.219 18.162.115.233 16.163.40.98 16.163.52.236 18.166.70.2 16.162.44.224 \
         16.163.52.8 18.163.101.162 18.162.59.48 16.163.42.228 18.167.68.65 18.166.54.234 18.167.102.139 16.163.55.96 43.198.23.137 18.166.51.62 18.166.57.230 18.166.63.170 18.167.96.86 18.162.143.131 18.162.114.212 18.166.65.64 18.166.63.182 43.198.74.139 43.198.73.182 16.163.57.14"
# clients="18.163.105.212 18.162.62.213 43.198.71.162 18.163.111.207 18.163.101.42 18.163.50.216 18.163.121.50 18.166.63.18 43.198.72.156 18.162.148.230 18.162.55.17 18.163.117.50 16.163.108.47 18.163.129.245 18.162.155.115 16.162.91.252 43.198.21.64 18.166.78.138 43.198.75.185 43.198.73.38"
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
# model="ResNet20_Cifar10"
# model="ResNet18_Cifar10"
# model="MobileNetV2_Cifar10"
model="LSTM_KWS"
# model="CNN_Cifar10"
# model="VGG11_Cifar10"
# model="CNN_Mnist"
# model="CNN_EMnist"

### dateset
dataset="KWS"
# dataset="Cifar10"
# dataset="EMnist"

### interlayer type
# interlayer_type="Default"
interlayer_type="CAS"

### sync setting
batchsize=20
init_period=100
dirichlet_alpha=0.05

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
        sleep 0.2
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
        sleep 0.2
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
        sleep 0.2
    done
fi




