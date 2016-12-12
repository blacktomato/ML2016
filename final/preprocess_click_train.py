import sys
import random
import numpy as np
from sklearn.utils import shuffle

with open(sys.argv[1] + '/clicks_train.csv', 'r') as fp:
    ofp = open(sys.argv[2] + 'clicks_train_small.csv', 'w+')
    lines = fp.readlines()

    line = lines[0]
    i = line.split(",")

    unclicks = []   # unclick lines
    selected = []   # lines to write in file
    prev_display = i[0]
    prev_click = i[2]
    ofp.write(line)

    for line_id in range(1,len(lines)):
        line = lines[line_id]
        i = line.split(",")
        
        if(int(i[2]) == 0):
            unclicks.append(line)
        else:
            selected.append(line)

        # if last of the same display_id
        if(line == lines[-1] or lines[line_id + 1].split(",")[0] != i[0]):
            if(len(unclicks) != 0):
                selected.append(random.choice(unclicks))
                
                # write in first selected line
                line_1 = random.choice(selected)
                ofp.write(line_1)
                # write in second selected line
                selected.remove(line_1)
                ofp.write(selected[0])
                # reset
                unclicks = []
                selected = []
            
            elif(len(selected) != 0):
                ofp.write(selected[0])
                selected = []

