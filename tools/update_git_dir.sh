cd ~
KEY_PEM_NAME=YourKeyName.pem
export DEEPLEARNING_WORKERS_COUNT=`wc -l < hosts`

sudo bash -c "cat hosts >> /etc/hosts"

for i in $(seq 2 $DEEPLEARNING_WORKERS_COUNT);
  do
  ssh -i ${KEY_PEM_NAME} deeplearning-worker${i} 'cd ~/pytorch_distributed_nn; git pull'
  echo "Done pull git repo on worker: deeplearning-worker${i}"
 done