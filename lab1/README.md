# FedML


### Experiment 1
```
python3 lab1.py \
--wandb_name fedavg_cifar10_adam_r20_test1 \
--gpu 1 \
--dataset cifar10 \
--data_dir ./cifar10 \
--client_num_in_total 20 \
--client_num_per_round 20 \
--comm_round 100 \
--frequency_of_the_test 1 \
--epochs 1 \
--batch_size 16 \
--client_optimizer adam \
--lr 0.001 \
--ci 1 \
--seed 123
```