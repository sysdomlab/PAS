#!/usr/bin/env bash
home=$HOME'/CASP-code'
clients='18.166.60.194 18.167.19.71 43.198.12.123 18.162.53.5 18.166.226.17 43.198.83.225 43.198.82.166 18.163.74.111 16.163.140.163 18.166.176.179 18.163.186.67 18.166.71.104 18.163.99.181 43.198.80.32 18.166.152.143 18.163.116.47 18.167.19.13 18.163.37.52 16.162.137.3 18.166.59.154'

log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

bandwidth=5120
num=0
for i in $clients; do
    # command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $bandwidth -d $bandwidth && speedtest"
    command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done