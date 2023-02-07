#!/usr/bin/env bash
home=$HOME'/CASP-code'
master="16.163.55.246"
workers="16.163.55.246 18.163.84.174 18.166.71.35 16.162.106.7 18.162.48.187 18.166.75.148 43.198.16.173 43.198.13.209 18.162.152.136 16.163.106.124 16.162.87.196 43.198.86.121 43.198.11.127 16.162.105.36 16.162.22.184 18.162.168.223 18.166.31.49 18.163.195.198 16.162.45.244 18.162.230.3"

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