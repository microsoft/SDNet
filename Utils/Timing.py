# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import datetime

timeelapsed = {}
startTime = {}
endTime = {}

def timerstart(name):
    startTime[name] = datetime.now()

def timerstop(name):
    endTime[name] = datetime.now()
    if not name in timeelapsed:
        timeelapsed[name] = endTime[name] - startTime[name]
    else:
        timeelapsed[name] += endTime[name] - startTime[name]

def timerreport():
    total = 0
    for name in timeelapsed:
        total += timeelapsed[name].total_seconds()
    print('')    
    print('----------------Timer Report----------------------')    
    for name, value in sorted(timeelapsed.items(), key = lambda item: -item[1].total_seconds()):
        print('%s: used time %s, %f%% ' % ('{:20}'.format(name), str(value).split('.')[0], value.total_seconds() / total * 100.0))
    print('--------------------------------------------------')    
    print('')
