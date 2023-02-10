#!/usr/bin/env bash
home=$HOME'/CASP-code'
workers='18.167.0.57 18.163.188.30 18.162.44.220 43.198.95.97 16.163.156.117 18.162.123.143 18.162.44.79 18.167.134.169 43.198.96.81 18.163.24.92 16.163.1.98 18.162.120.27 18.167.51.97 43.198.95.178 18.162.43.201 18.163.194.43 18.162.142.216 16.162.88.146 43.198.90.67 18.167.165.118'

log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

bandwidth=5120
num=0
for i in $workers; do
    command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $bandwidth -d $bandwidth && speedtest"
    # command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done