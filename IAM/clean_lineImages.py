import os
import glob

for filepath in glob.iglob('lineImages/**/**/*..png'):
    os.remove(filepath)
    print(filepath, ' Removed!')