#!/usr/bin/env bash
home=$HOME'/CASP-code'
clients='43.198.12.27 43.198.79.213 18.166.59.134 18.163.120.150 18.163.184.182 43.198.81.171 16.162.109.47 18.162.210.241 18.166.30.27 43.198.86.185 18.167.19.119 16.162.122.212 16.163.144.7 16.163.147.237 18.166.75.251 43.198.17.53 18.163.119.213 16.162.143.92 18.162.154.1 16.162.163.97'

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