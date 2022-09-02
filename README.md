# adversarial_attack
eps = [0.1, 0.3, 0.7, 1e-2, 3e-2, 1e-3] 
to modify this, go to main.py Line 31 to change this line. 

* widen-factor = 1, 2, 4, 8, 16 
* epochs = 300 by default
* attack_method = FGSM, PGD, FAB
* evaluate_frequency = default 50 (how many epochs to evaluate the robustness)
* (optional: attack_param = method specific. no need to put anything here for our experiment. 
the noise level eps are preset in main.py)
* (optional: manualSeed, if not specified, 
a random number from 0 - 10000 will be assigned to be the seed)

output file: 
checkpoint/<dataset>/wrn_D<depth>_W<widen-factor>_<attack_method>/randSeed

CIFAR10: 
`python main.py --dataset cifar10 --depth 28 --widen-factor <> --epochs <> --attack_method <> --evaluate_frequency <>`

EMNIST: 
`python main.py --dataset emnist --depth 16 --widen-factor <> --epochs <> --attack_method <> --evaluate_frequency <>`



#################################################
main2.py : easier to handle preemption on vector

always keep the resume flag true. It will check whether the provided manualSeed already exists. 
if exists, resume. else create a new run. 
* experiment_name = (your experiment name)
CIFAR10: 
`python main2.py --dataset cifar10 --depth 28 --widen-factor <> --epochs <> --attack_method <> --evaluate_frequency <> --resume True --manualSeed <> --experiment_name <>`

EMNIST: 
`python main2.py --dataset emnist --depth 16 --widen-factor <> --epochs <> --attack_method <> --evaluate_frequency <> --resume True --manualSeed <> --experiment_name <>`

########################################################
main-together.py: evaluate all attacks together
```
noise_levels = {'PGD': [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.3 ],
                'FGSM': [0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03],
                'GN': [0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03]}

list_attack_methods = ['PGD', 'FGSM', 'GN']
```

sample run:
```
python main-together.py --dataset emnist --depth 16 --epochs 150 --lr 0.03 --schedule 75 110 --widen-factor ${width} --resume True --manualSeed ${seed} --experiment_name <>
python main-together.py --dataset fashionemnist --depth 16 --epochs 150 --lr 0.03 --schedule 75 110 --widen-factor ${width} --resume True --manualSeed ${seed} --experiment_name <>
python main-together.py --dataset cifar10 --depth 28 --widen-factor ${width} --resume True --manualSeed ${seed} --experiment_name <> --evaluate_frequncy 100

```
