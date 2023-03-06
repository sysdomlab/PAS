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

clients='18.162.116.187 18.162.60.144 18.162.61.61 18.166.52.231 18.163.181.19 16.163.57.54 18.162.126.102 18.162.125.252 43.198.76.243 18.162.193.169 18.163.105.64 18.162.193.249 18.162.56.116 43.198.73.46 18.166.76.162 16.163.108.79 43.198.71.79 18.167.84.223 16.163.109.139 18.162.151.153'
# clients='43.198.24.242 43.198.72.11 43.198.77.77 43.198.22.171'
for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done