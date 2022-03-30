import os
root_dir = '/home/photon/Datasets/Libri2Mix/wav16k/mix_single_both'
subdirs = ['test', 'dev', 'train-360']
for subdir in subdirs:
    subdirpath = root_dir + '/' + subdir
    for s in ['s1', 's2', 'mix_both']:
        spath = subdirpath + '/' + s
        for f in os.listdir(spath):
            if len(f.split('_')) <= 2: 
                # os.system('rm {}'.format(spath + '/' + f))
                print(f)
            