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

clients='18.166.67.81 18.166.53.234 18.162.214.77 18.166.58.223 16.163.126.159 16.163.103.144 16.163.55.148 43.198.78.224 18.167.96.106 18.163.123.17 18.163.77.138 43.198.71.113 18.166.51.241 18.166.55.213 18.167.121.232 18.167.85.67 18.166.55.177 43.198.76.107 18.163.8.239 18.163.35.255'
# clients='43.198.24.242 43.198.72.11 43.198.77.77 43.198.22.171'
for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done