#!/usr/bin/env bash
home=$HOME'/CASP-code'
clients='16.162.24.129 18.166.62.132 43.198.86.93 16.162.34.42 43.198.14.118 18.162.112.54 18.166.226.217 16.163.149.8 18.162.149.135 16.162.45.172 16.163.142.94 18.166.29.120 18.163.114.13 18.162.112.109 16.163.155.86 43.198.11.132 16.162.118.131 16.163.149.227 18.166.215.47 16.162.119.231'

log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

bandwidth=1024
num=0
for i in $clients; do
    # command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $bandwidth -d $bandwidth && speedtest"
    command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done