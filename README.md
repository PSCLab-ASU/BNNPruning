## Pruning BNN

This is the source code for paper BNN Pruning: Pruning Binary Neural Network Guided by Weight
Flipping Frequency (https://ren-fengbo.lab.asu.edu/sites/default/files/pid6334845.pdf). The paper is accepted by 21st IEEE International Symposium on Quality Electronic Design (ISQED).

The code is written based upon jiecaoyu/XNOR-Net-PyTorch (https://github.com/jiecaoyu/XNOR-Net-PyTorch). Please follow XNOR-Net-PyTorch to setup the environment and place the modified files in the CIFAR-10 folder. 

## Training
run main.py to start training. It can log the weight flipping matrices during training.

## Post-training 
run run.py to analysis weight flipping frequency.
