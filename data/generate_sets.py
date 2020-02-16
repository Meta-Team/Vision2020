import os
import random
 
total_jpg = os.listdir('images')
total_name = ['data/images/'+i+'\n' for i in total_jpg]
 
ftrain = open('train.txt', 'w')
fvalid = open('valid.txt', 'w')
 
for i in total_name:
    if random.random() > 0.7:
        ftrain.write(i)
    else:
        fvalid.write(i)
 
ftrain.close()
fvalid.close()
