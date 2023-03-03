#!/usr/bin/env bash
home=$HOME'/CASP-code'

clients='18.163.112.214 18.162.149.191 16.163.141.88 16.162.45.130 18.166.225.51 18.162.111.137 16.163.146.64 18.162.111.65 18.163.182.179 18.166.178.80 18.166.30.86 43.198.86.162 18.163.182.123 18.162.154.99 18.163.124.20 16.163.143.211 18.163.114.212 16.162.45.177 18.167.20.19 18.166.62.148'
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