#!/usr/bin/env bash
home=$HOME'/CASP-code'
clients='43.198.12.27 43.198.79.213 18.166.59.134 18.163.120.150'

log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

bandwidth=5120
num=0
for i in $clients; do
    command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $bandwidth -d $bandwidth && speedtest"
    # command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done