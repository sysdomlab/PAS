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

clients='43.198.24.242 43.198.72.11 16.163.43.98 18.167.90.183 18.163.123.55 18.163.127.201 18.166.78.71 18.166.78.90 43.198.77.77 18.162.155.215 18.163.111.72 18.162.214.100 43.198.22.171 16.163.58.206 18.167.103.202 18.163.50.244 18.167.103.113 18.162.214.8 18.163.73.23 18.166.78.83'
for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done