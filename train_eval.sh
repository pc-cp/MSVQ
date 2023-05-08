nohup python msvq.py --mk 0.99 --mp 0.95 --dataset cifar10 --tem 0.04 --weak --gpuid 0 --logdir cifar10_00 >stdout/cifar10_00 2>&1 &
nohup python msvq.py --mk 0.99 --mp 0.95 --dataset cifar100 --tem 0.04 --weak --gpuid 1 --logdir cifar100_00 >stdout/cifar100_00 2>&1 &
nohup python msvq.py --mk 0.996 --mp 0.99 --dataset stl10 --tem 0.04 --weak --gpuid 1 --logdir stl10_00 >stdout/stl10_00 2>&1 &
nohup python msvq.py --mk 0.996 --mp 0.99 --dataset tinyimagenet --tem 0.04 --weak --gpuid 1 --logdir tinyimagenet_00 >stdout/tinyimagenet_00 2>&1 &

nohup python linear_eval.py --dataset cifar10  --gpuid 0  --logdir cifar10_00 >stdout/cifar10_00_01 2>&1 &
nohup python linear_eval.py --dataset cifar100  --gpuid 0  --logdir cifar100_00 >stdout/cifar100_00_01 2>&1 &
nohup python linear_eval.py --dataset stl10  --gpuid 0  --logdir stl10_00 >stdout/stl10_00_01 2>&1 &
nohup python linear_eval.py --dataset tinyimagenet  --gpuid 0  --logdir tinyimagenet_00 >stdout/tinyimagenet_00_01 2>&1 &