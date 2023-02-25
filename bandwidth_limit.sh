#!/usr/bin/env bash
home=$HOME'/CASP-code'
clients='16.162.102.20 18.162.168.86 43.198.86.242 18.166.154.157 16.162.100.184 16.163.138.198 18.166.31.181 18.162.52.54 18.162.49.84 16.162.188.65 18.166.31.175 18.163.37.39 43.198.85.223 18.166.214.136 43.198.86.8 18.167.36.91 16.163.155.174 16.162.163.239 16.162.163.160 18.162.152.107'

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