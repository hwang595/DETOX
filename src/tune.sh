tune_dir=${HOME}/DETOX/src/tune/
max_tuning_step=200
method=bulyan
mkdir ${tune_dir}

echo "Start parameter tuning ..."
for lr in 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125
do
  echo "Trial running for learning rate: ${lr}"
  mpirun -n 46 --hostfile hosts_address \
  python distributed_nn.py \
  --lr=${lr} \
  --momentum=0.9 \
  --network=ResNet18 \
  --dataset=Cifar10 \
  --batch-size=32 \
  --comm-type=Bcast \
  --mode=${method} \
  --approach=baseline \
  --eval-freq=${max_tuning_step} \
  --err-mode=constant \
  --adversarial=-100 \
  --epochs=50 \
  --max-steps=${max_tuning_step} \
  --worker-fail=5 \
  --group-size=3 \
  --compress-grad=compress \
  --bucket-size=5 \
  --checkpoint-step=0 \
  --lis-simulation=simulate \
  --train-dir=/home/ubuntu/ > ${tune_dir}${method}_lr_${lr} 2>&1

  cat ${tune_dir}${method}_lr_${lr} | grep Step:\ ${max_tuning_step} > ${tune_dir}${method}_lr_${lr}_processing
  bash ../tools/killall.sh
done

for lr in 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125
do
  echo "Logging out tunning results"
  python tiny_tuning_parser.py \
  --tuning-dir=${tune_dir}${method}_lr_${lr}_processing \
  --tuning-lr=${lr} \
  --num-workers=45
done
