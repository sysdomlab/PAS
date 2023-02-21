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
ps -ef | grep "multiprocessing" | awk '{print $2}' | xargs kill -9


clients='18.162.55.161 18.166.67.172 18.163.77.191 16.163.40.18 18.163.50.185 18.162.116.86 18.163.73.147 18.162.114.6 43.198.24.254 16.163.103.16 18.163.129.160 43.198.74.74 18.163.8.137 18.166.78.151 18.166.54.95 18.167.109.223 18.166.51.117 18.163.121.81 43.198.19.64 18.163.123.206'

for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
	command="ps -ef | grep 'multiprocessing' | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done