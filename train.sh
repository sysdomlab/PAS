#!/usr/bin/env bash
### set home directory
home=$HOME'/CASP-code'

### prepare master and workers
server='localhost'
# server='16.162.205.1'
# clients="18.167.116.153 43.198.22.242 43.198.78.249 18.162.155.168 18.162.245.41 18.162.146.158 16.163.109.65 18.163.77.129 43.198.25.218 18.162.148.141 43.198.71.89 43.198.74.157 43.198.74.91 43.198.76.241 16.163.40.222 18.166.52.232 18.166.56.24 18.162.126.123 18.167.102.143 18.166.70.66 \
#          18.162.57.220 18.162.57.186 18.163.127.196 18.162.114.16 16.163.108.119 18.167.72.188 16.163.108.5 18.166.55.165 43.198.77.149 18.162.60.64 18.163.123.14 18.166.53.246 18.162.114.124 18.166.51.81 16.162.91.210 43.198.71.167 18.166.70.200 18.162.155.181 16.162.25.44 43.198.22.222 \
#          18.162.115.160 18.162.146.109 18.162.55.224 18.163.192.216 16.163.52.255 18.163.105.94 18.166.74.82 43.198.77.138 18.166.78.131 43.198.23.45 18.166.63.52 43.198.24.179 16.163.58.49 18.166.55.85 16.162.44.130 18.166.76.41 16.162.91.105 16.163.42.84 18.163.73.211 18.162.62.57 \
#          16.163.41.206 18.163.180.53 18.163.180.235 18.167.102.138 18.162.115.17 18.166.78.43 18.166.57.118 43.198.72.9 16.163.56.87 18.163.187.237 43.198.25.110 18.167.94.121 18.162.245.158 18.162.143.222 18.166.76.228 18.163.117.3 18.166.58.5 18.163.121.207 18.162.245.185 16.163.109.174 \
#          18.163.35.74 18.166.78.255 18.163.77.142 18.163.183.206 18.162.61.209 18.166.56.177 18.163.50.161 43.198.24.28 18.162.114.184 18.162.214.69 18.162.62.228 18.162.57.238 18.163.77.126 18.163.181.197 16.163.128.254 43.198.25.115 16.163.103.170 43.198.74.84 18.166.77.62 18.162.56.77"
# clients="18.163.105.212 18.162.62.213 43.198.71.162 18.163.111.207 18.163.101.42 18.163.50.216 18.163.121.50 18.166.63.18 43.198.72.156 18.162.148.230 18.162.55.17 18.163.117.50 16.163.108.47 18.163.129.245 18.162.155.115 16.162.91.252 43.198.21.64 18.166.78.138 43.198.75.185 43.198.73.38"
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
interlayer_type="Default"
# interlayer_type="CAS"

### sync setting
batchsize=50
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




