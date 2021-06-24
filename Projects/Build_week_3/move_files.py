import glob
import os
import re
import shutil

def move_to(user_name, from_, to):

    cls = ['dog', 'fish', 'rabbit']

    for c in cls:
        files = glob.glob('images/' + from_ + '/' + c + '/*.png' )
        for file in files:
            pattern = user_name + '.png'
            if(re.findall("%s" % pattern, file)):
                source = 'images/' + from_+ '/' + file + '/' + c
                dest = 'images/' + to + '/' + c
                print('moving file from {} to {}'.format(source, dest))
                shutil.move(file, dest)

# params are user_name, from where to want to move and where to
move_to('Dalli', 'validation', 'train')