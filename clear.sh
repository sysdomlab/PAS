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

clients='18.166.67.56 16.163.58.32 18.166.65.15 18.167.96.144 18.166.77.173 18.167.109.117 16.163.56.102 18.166.67.59 18.166.70.7 18.162.126.68 18.163.5.183 18.166.67.217 16.163.109.65 18.162.116.225 43.198.77.24 18.167.102.181 18.162.143.142 18.163.181.40 18.167.116.234 18.163.105.192'
# clients='43.198.24.242 43.198.72.11 43.198.77.77 43.198.22.171'
for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done