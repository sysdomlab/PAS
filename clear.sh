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

clients='16.162.55.42 18.163.8.28 43.198.75.124 18.166.67.79 18.166.58.104 18.162.225.50 43.198.75.203 18.163.115.32 18.163.185.130 18.163.73.40 18.166.57.113 18.166.67.234 16.162.55.172 18.163.103.45 18.163.129.100 18.162.125.60 18.166.53.177 18.167.116.162 16.162.44.164 18.162.225.253'
# clients='43.198.24.242 43.198.72.11 43.198.77.77 43.198.22.171'
for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done