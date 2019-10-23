# File to generate the gtFile needed by the create_lmdb_dataset.py script
# Format is : /path/to/image.png Transcription

import pandas as pd
import numpy as np

def main():
    trainset = pd.read_csv('trainset.txt')
    testset_t = pd.read_csv('testset_t.txt')
    testset_v = pd.read_csv('testset_v.txt')

    trainpaths = set_to_path(trainset)
    testpaths_t = set_to_path(testset_t)
    testpaths_v = set_to_path(testset_v)
    
    # Some hacky stuff because pandas sucks sometimes
    line_list = []
    with open('wordImages/words.txt') as f:
        line_list = f.readlines()
    line_list = [[x.split(' ')[0], x.split(' ')[len(x.split(' '))-1]] for x in line_list if x[0] != '#']
    line_list = [[x[0], x[1][:-1]] for x in line_list]
    words = np.array(line_list)
    words[:, 0] = word_to_path(words[:, 0])
    path_pieces = words[:, 0]
    path_pieces = [x.split('-') for x in path_pieces]
    words[:, 0] = path_pieces_to_path(path_pieces)
    

    match_traintest = []
    match_traintest = [x.split('/')[0] + '/' + x.split('/')[1] + '/'  for x in words[:, 0]]
    
    trainpaths = np.array(trainpaths).flatten()
    testpaths_t = np.array(testpaths_t).flatten()
    testpaths_v = np.array(testpaths_v).flatten()
    match_traintest = np.array(match_traintest).flatten()
    
    train_ids = getids(trainpaths, match_traintest)
    testt_ids = getids(testpaths_t, match_traintest)
    testv_ids = getids(testpaths_v, match_traintest)

    print (train_ids, testt_ids, testv_ids)

    print(words)

    # Now print out the files

    with open('train_gt.txt', 'w') as f:
        for item in words[train_ids].tolist():
            f.write(item[0] + ' ' + item[1] + '\n')
    with open('testt_gt.txt', 'w') as f:
        for item in words[testt_ids].tolist():
            f.write(item[0] + ' ' + item[1] + '\n')
    with open('testv_gt.txt', 'w') as f:
        for item in words[testv_ids].tolist():
            f.write(item[0] + ' ' + item[1] + '\n')

def getids(paths, match_traintest):
    idxs = [np.nonzero(np.where(match_traintest==x, 1, 0)) for x in paths]
    idxs = np.array(idxs).flatten()
    idxs = np.concatenate(idxs).ravel()
    return idxs



def path_pieces_to_path(path_pieces):
    paths = [x[0] + '/' + x[0] + '-' + x[1] + '/' + x[0]+'-'+x[1]+'-'+x[2]+'-'+x[3] for x in path_pieces]
    return paths

def word_to_path(labels):
    labels = [x + '.png' for x in labels]
    return labels

def set_to_path(df):
    vals =list(df.to_numpy().flatten())
    paths = [x.split('-')[0] + '/'  + x.split(' ')[1] + '/' for x in vals]
    paths = [x.split(' ')[1] for x in paths]
    return np.array(paths)
    

if __name__ == '__main__':
    main()
