if [ -z "$1" ]
then
		echo 'Please input the exp trial_no to kill!'
		echo
		echo 'Ongoing trial processes for your reference: '
		ps -ef | grep `pwd` | grep 'ssh' | grep 'rank=0'
		exit
fi
echo 'Process killed: '
ps -ef | grep `pwd` | grep "[t]rial_no=$1"
ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print $2}' | xargs kill -9

clients='18.162.111.206 18.167.37.77 18.163.116.89 18.162.51.27 18.163.193.171 43.198.14.29 18.162.130.197 18.162.112.14 43.198.86.202 18.166.176.33 18.162.53.200 18.163.120.198 18.162.149.33 18.163.7.50 18.166.236.8 18.163.112.220 18.166.113.56 18.162.54.255 18.163.114.3 16.163.147.104'
for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done