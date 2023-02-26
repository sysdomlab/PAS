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

clients='43.198.12.27 43.198.79.213 18.166.59.134 18.163.120.150 18.163.184.182 43.198.81.171 16.162.109.47 18.162.210.241 18.166.30.27 43.198.86.185 18.167.19.119 16.162.122.212 16.163.144.7 16.163.147.237 18.166.75.251 43.198.17.53 18.163.119.213 16.162.143.92 18.162.154.1 16.162.163.97'
for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done