import pandas as pd
noise_levels = [0.1, 0.3, 0.7, 1e-2, 3e-2, 1e-3]
names = ['epoch', 'Learning Rate', 'Train Loss', 'Valid Loss',
                  'Train Acc.', 'Valid Acc.',
                  'Train topk acc.', 'Valid topk acc.']
for nl in noise_levels:
    names += [f'rtest_loss-{nl}', f'rtest_acc-{nl}', f'rtest_acc_topk-{nl}']

pd.read_csv('xxx/log.txt', sep='\t', names=names)