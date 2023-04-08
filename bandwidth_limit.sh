#!/usr/bin/env bash
home=$HOME'/CASP-code'

clients="18.162.188.244 18.162.194.201 16.163.94.54 18.167.167.45"
# clients='43.198.24.242 43.198.72.11 43.198.77.77 43.198.22.171'
log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

up_bandwidth=1024
down_bandwidth=1024
num=0
for i in $clients; do
    command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $up_bandwidth -d $down_bandwidth && speedtest"
    # command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    # command="speedtest"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done