#!/usr/bin/env bash
home=$HOME'/CASP-code'

clients='18.166.55.130 18.167.85.7 18.163.111.195 18.166.63.24 16.162.55.143 43.198.71.196 18.167.68.174 18.163.35.5 18.162.214.72 16.163.57.106 18.163.111.48 18.167.103.121 18.166.55.75 43.198.71.241 16.163.52.171 18.162.134.98 16.163.42.208 18.166.65.224 18.166.74.175 16.162.86.7'
log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

bandwidth=1024
num=0
for i in $clients; do
    command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $bandwidth -d $bandwidth && speedtest"
    # command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done