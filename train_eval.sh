nohup python main.py --name msvq --m1 0.99  --m2 0.95 --dataset cifar10       --tem 0.04 --weak --gpuid 0 --logdir cifar10_00 >stdout/cifar10_00 2>&1 &
nohup python main.py --name msvq --m1 0.99  --m2 0.93 --dataset cifar100      --tem 0.03 --weak --gpuid 0 --logdir cifar100_00 >stdout/cifar100_00 2>&1 &
nohup python main.py --name msvq --m1 0.996 --m2 0.99 --dataset stl10         --tem 0.04 --weak --gpuid 0 --logdir stl10_00 >stdout/stl10_00 2>&1 &
nohup python main.py --name msvq --m1 0.996 --m2 0.99 --dataset tinyimagenet  --tem 0.04 --weak --gpuid 0 --logdir tinyimagenet_00 >stdout/tinyimagenet_00 2>&1 &

nohup python linear_eval.py --name msvq --dataset cifar10       --gpuid 0  --logdir cifar10_00 >stdout/cifar10_00_01 2>&1 &
nohup python linear_eval.py --name msvq --dataset cifar100      --gpuid 0  --logdir cifar100_00 >stdout/cifar100_00_01 2>&1 &
nohup python linear_eval.py --name msvq --dataset stl10         --gpuid 0  --logdir stl10_00 >stdout/stl10_00_01 2>&1 &
nohup python linear_eval.py --name msvq --dataset tinyimagenet  --gpuid 0  --logdir tinyimagenet_00 >stdout/tinyimagenet_00_01 2>&1 &