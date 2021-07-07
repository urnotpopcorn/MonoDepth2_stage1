import sys
import os

input_dir = sys.argv[1]

algos = ['all', 'fg', 'bg']
for i in range(20):
    index = str(i)
    test_path = os.path.join(input_dir, algos[0], 'weights_'+index+'.log')
    if os.path.exists(test_path) == False:
        continue
    print('weights_'+index, end='\t')
    
    try:
        for algo in algos:
            file_path = os.path.join(input_dir, algo, 'weights_'+index+'.log')
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if 'Noc EPE' in line:
                        noc = line.split()[-1]
                    elif 'Occ EPE' in line:
                        occ = line.split()[-1]

            print(noc, occ, sep='\t', end='\t')
    except:
        pass

    print()    



