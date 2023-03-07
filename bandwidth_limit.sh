#!/usr/bin/env bash
home=$HOME'/CASP-code'

clients='16.162.55.42 18.163.8.28 43.198.75.124 18.166.67.79 18.166.58.104 18.162.225.50 43.198.75.203 18.163.115.32 18.163.185.130 18.163.73.40 18.166.57.113 18.166.67.234 16.162.55.172 18.163.103.45 18.163.129.100 18.162.125.60 18.166.53.177 18.167.116.162 16.162.44.164 18.162.225.253'
# clients='43.198.24.242 43.198.72.11 43.198.77.77 43.198.22.171'
log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

up_bandwidth=1024
down_bandwidth=1280
num=0
for i in $clients; do
    command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $up_bandwidth -d $down_bandwidth && speedtest"
    # command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    # command="speedtest"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done