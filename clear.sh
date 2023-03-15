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

# clients='18.166.56.96 18.163.121.196 18.163.129.10 43.198.25.89 18.163.117.175 18.163.181.66 18.167.120.50 18.163.77.224 18.163.115.91 43.198.74.77 18.167.72.190 16.163.108.138 18.166.74.65 16.162.44.67 16.162.48.158 18.167.116.245 43.198.77.182 16.163.57.59 18.163.123.184 18.163.190.232'
# # clients='43.198.24.242 43.198.72.11 43.198.77.77 43.198.22.171'
# for i in $clients; do
# 	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
# 	ssh $i $command
# done