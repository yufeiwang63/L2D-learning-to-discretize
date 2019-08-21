'''
used to clean all unnecssary pytorch saved models.
but preserve the log file.
'''

import sys
import os

dirs = os.listdir('../../')
dirs = sorted(dirs)

for dir in dirs:
    print(dir)
    if dir[0:8] != 'Burgers-':
        continue
    if dir >= 'Burgers-2019-02-25-22-28-39':
        break 
    
    ### if there is a vedio in the dir, then this dir is important, should not delete it.
    files = os.listdir('../' + dir + '/')
    video = False
    for f in files:
        if f.find('mp4') != -1:
            video = True
            break

    if video:
        continue

    ### delete all the saved pytorch models. Only store the log file.
    files = os.listdir('../' + dir + '/')
    for file in files:
        print(file)
        if file != 'log.txt':
            os.remove('../' + dir + '/' + file)

    
    
