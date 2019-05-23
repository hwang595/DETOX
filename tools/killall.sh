KEY_PEM_NAME=${HOME}/.ssh/XXXXXXXXXX.pem
export DEEPLEARNING_WORKERS_COUNT=`wc -l < hosts`

for i in $(seq 2 $DEEPLEARNING_WORKERS_COUNT);
  do
  ssh -i ${KEY_PEM_NAME} deeplearning-worker${i} 'killall python'
 done