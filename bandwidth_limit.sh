#!/usr/bin/env bash
home=$HOME'/CASP-code'

clients="43.198.101.204 16.162.252.241 18.162.133.190 18.162.156.192 43.198.107.219 18.163.198.14 16.163.105.146 16.162.21.106 43.198.88.158 18.162.123.234 18.166.61.2 43.198.101.156 18.163.46.141 18.167.166.184 18.163.210.25 18.163.198.94 43.198.102.7 18.167.59.170 16.163.94.64 18.163.2.50 \
         43.198.90.182 18.162.191.3 18.162.41.90 16.163.98.205 18.163.40.184 18.163.214.108 43.198.107.75 16.162.161.190 18.162.194.13 18.163.2.48 43.198.101.241 18.163.238.15 18.162.45.54 18.162.232.193 18.167.166.91 16.163.104.74 18.163.214.146 43.198.104.83 16.163.94.14 43.198.101.233"
# clients='18.166.73.214 18.162.42.252 18.162.228.74 18.167.99.138'
log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

up_bandwidth=1024
down_bandwidth=1024
num=0
for i in $clients; do
    command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $up_bandwidth -d $down_bandwidth && speedtest"
    # command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    # command="speedtest"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done