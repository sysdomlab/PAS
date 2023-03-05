#!/usr/bin/env bash
home=$HOME'/CASP-code'

clients='18.162.111.206 18.167.37.77 18.163.116.89 18.162.51.27 18.163.193.171 43.198.14.29 18.162.130.197 18.162.112.14 43.198.86.202 18.166.176.33 18.162.53.200 18.163.120.198 18.162.149.33 18.163.7.50 18.166.236.8 18.163.112.220 18.166.113.56 18.162.54.255 18.163.114.3 16.163.147.104'
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