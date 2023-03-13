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

clients='16.162.91.136 16.163.56.175 18.163.102.176 16.163.58.212 18.163.115.244 18.167.96.212 18.163.123.97 18.166.72.149 18.162.245.130 18.167.94.162 18.166.65.116 18.167.84.185 18.162.62.94 43.198.72.223 43.198.77.39 18.167.121.223 18.163.123.149 18.167.121.144 18.162.126.139 18.162.114.30'
# clients='43.198.24.242 43.198.72.11 43.198.77.77 43.198.22.171'
for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done