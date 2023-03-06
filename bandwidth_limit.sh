#!/usr/bin/env bash
home=$HOME'/CASP-code'

clients='18.162.116.187 18.162.60.144 18.162.61.61 18.166.52.231 18.163.181.19 16.163.57.54 18.162.126.102 18.162.125.252 43.198.76.243 18.162.193.169 18.163.105.64 18.162.193.249 18.162.56.116 43.198.73.46 18.166.76.162 16.163.108.79 43.198.71.79 18.167.84.223 16.163.109.139 18.162.151.153'
# clients='43.198.24.242 43.198.72.11 43.198.77.77 43.198.22.171'
log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

bandwidth=1024
num=0
for i in $clients; do
    # command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $bandwidth -d $bandwidth && speedtest"
    # command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    command="speedtest"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done