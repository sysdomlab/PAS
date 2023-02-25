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

clients='18.166.60.194 18.167.19.71 43.198.12.123 18.162.53.5 18.166.226.17 43.198.83.225 43.198.82.166 18.163.74.111 16.163.140.163 18.166.176.179 18.163.186.67 18.166.71.104 18.163.99.181 43.198.80.32 18.166.152.143 18.163.116.47 18.167.19.13 18.163.37.52 16.162.137.3 18.166.59.154'
for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done