#!/usr/bin/env bash
home=$HOME'/CASP-code'
master="18.162.155.234"
workers="18.162.155.234 18.162.45.16 18.163.116.86 18.166.176.205 16.162.47.226 18.162.154.167 43.198.13.129 18.163.56.89 18.162.230.55 16.163.140.148 43.198.15.247 18.166.29.93 16.162.109.34 16.163.129.15 18.162.110.27 18.163.112.138 18.162.47.90 18.162.167.16 43.198.81.107 18.163.7.117"

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