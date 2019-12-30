# File to generate the gtFile needed by the create_lmdb_dataset.py script
# Format is : /path/to/image.png Transcription

'''
# TODO: The stupid fucking numbers sometimes start at 01 and other times start at 00 so 
        I'm gonna have to figure out some way to fix that. I'm so sick of trying to clean
        up this trash ass dataset.
'''
import pandas as pd
import numpy as np
import os
import cv2
import glob

def binarize(img, gv):
    ret, bin_img = cv2.threshold(img, gv, 255, cv2.THRESH_BINARY)
    return bin_img

def main():

    # Data is split as follows:
    #   training - standard offline (6161) + val2 + everthing else in online.
    #   val1 - standard offline (900) + overlap in online data.
    #   val2 - does not exist, put into the training data.
    #   test - standard offline (1861) + overlap in online data.


    # Read in the data split id's
    gt_path = './largeWriterIndependentTextLineRecognitionTask/'
    trainset = pd.read_csv(os.path.join(gt_path, 'trainset.txt'))
    testset = pd.read_csv(os.path.join(gt_path, 'testset.txt'))
    valset1 = pd.read_csv(os.path.join(gt_path, 'validationset1.txt'))
    valset2 = pd.read_csv(os.path.join(gt_path, 'validationset2.txt'))
    trainset = trainset.to_numpy().flatten().tolist()
    testset = testset.to_numpy().flatten().tolist()
    valset1 = valset1.to_numpy().flatten().tolist()
    valset2 = valset2.to_numpy().flatten().tolist()

    # Create list of lines in the ground truth
    lines = []
    with open('./IAM_ascii/lines.txt') as f:
        for line in f:
            if line[0] != '#':
                lines.append(line)
    
    # Parse out data paths and grey values to convert them to greyscale
    path_grey_trans = []
    for line in lines:
        words = line.split(' ')
        path0 = words[0]
        split_path = path0.split('-')
        path_prefix1 = split_path[0]
        path_prefix2 = split_path[1]
        offline_path = './IAM_lines/' + path_prefix1 + '/' + path_prefix1 + '-' + path_prefix2 + '/' + path0 + '.png'
        grey_val = words[2]
        transcription_spaces = words[8:]
        pipetrans = transcription_spaces[0]
        for i in range(len(transcription_spaces)-1):
            pipetrans = pipetrans + ' ' + transcription_spaces[i+1]
        pipetrans = pipetrans.split('|')
        trans = pipetrans[0]
        for i in range(len(pipetrans)-1):
            trans = trans + ' ' + pipetrans[i+1]
        path_grey_trans.append((offline_path, path0, grey_val, trans))

    
    # Create a list of all the split's paths, grey values, and transcriptions.
    train_gt = []
    test_gt = []
    val1_gt = []
    val2_gt = []
    max_trans = 0
    for offline_path, path0, grey_val, trans in path_grey_trans:
        if len(trans) > max_trans:
            max_trans = len(trans)
        if path0 in valset1:
            val1_gt.append((offline_path, trans))
        elif path0 in valset2: # Note that these values are actually supposed to be included in training data as coded.
            train_gt.append((offline_path, trans))
        elif path0 in testset:
            test_gt.append((offline_path, trans))
        elif path0 in trainset:
            train_gt.append((offline_path, trans))
    # print(max_trans)

    # Add the paths of the online data to the training set and val1 set in 80/20 split
    online_list = []
    for file in glob.iglob('./lineImages/**/**/*.png'):
        online_list.append(file)
    
    # Create a dictionary linking names to paths for online data
    name_path_dict = {}
    for path in online_list:
        words = path.split('/')
        name_png = words[4]
        name = name_png[:-4]
        name_path_dict.update({name:path})

    # # Get all the image "names" and transcriptions in dictionary
    # name_trans_dict = {}
    # for line in lines:
    #     words = line.split(' ')
    #     img_name = words[0]
    #     transcription_spaces = words[8:]
    #     pipetrans = transcription_spaces[0]
    #     for i in range(len(transcription_spaces)-1):
    #         pipetrans = pipetrans + ' ' + transcription_spaces[i+1]
    #     pipetrans = pipetrans.split('|')
    #     trans = pipetrans[0]
    #     for i in range(len(pipetrans)-1):
    #         trans = trans + ' ' + pipetrans[i+1]
    #     name_trans_dict.update({img_name:trans})
    
    # The transcriptions are all fucked up. Getting them from CSR in the ASCII file
    name_transpath_dict={}
    for name, path in name_path_dict.items():
        transcription_path = path[:-7]
        transcription_path = transcription_path.replace('lineImages', 'ascii')
        path_words = transcription_path.split('/')
        path_last_word = path_words[-1]
        transcription_path = transcription_path+ '.txt' #[:-1] + '/' + path_last_word[:-1] 
        trans_name = name[:-3]
        if trans_name in name_transpath_dict.keys():
            name_transpath_dict[trans_name].append(transcription_path)
        else:
            name_transpath_dict.update({trans_name:[transcription_path]})
    for name in name_transpath_dict.keys():
        lines = []
        # Read in the ground truth file
        with open(name_transpath_dict[name][0]) as f:
            for line in f:
                lines.append(line)
        # Make sure we read from the CSR section
        in_csr = False
        counter = 0
        for line in lines:
            if line[:4] == 'CSR:':
                in_csr = True
            # if line[0] == '#' and line[1] == '\n' and name != 'h04-064':
            #         print(line)
            #         print(name)
            #         exit()
            if in_csr and line[:4] != 'CSR:' and line[0] != '\n':
                print(name)
                name_transpath_dict[name][counter] = line
                counter += 1
    path_trans_dict = {}
    for name in name_path_dict.keys():
        img_path = name_path_dict[name]
        img_name = name[:-3]
        trans_list = name_transpath_dict[img_name]
        path_trans_dict.update({img_path:trans_list[0]})
        name_transpath_dict[img_name].pop(0)
    
    online_path_trans_dict = {}
    for path in path_trans_dict.keys():
        path_words = path.split('/')
        name = path_words[4][:-4]
        on_off_line = path_words[1]
        if name in valset1:
            val1_gt.append((path, path_trans_dict[path]))
        elif name in testset:
            test_gt.append((path, path_trans_dict[path]))
        elif on_off_line == 'lineImages':
            online_path_trans_dict.update({path:path_trans_dict[path]})
        else:
            train_gt.append((path, path_trans_dict[path]))
    
    all_online_keys = list(online_path_trans_dict.keys())
    ordered_online_paths = []
    prev_name = ''
    counter = 0
    for key in all_online_keys:
        key_words = key.split('/')
        name = key_words[4][:-6]
        if name == prev_name:
            numbered_name = name + format(counter, '02d')
            counter += 1
        else:
            counter = 0
            numbered_name = name + '00'
        prev_name = name
        numbered_name = numbered_name + '.png'
        lead_path = ''
        for word in key_words[:-1]:
            lead_path = lead_path + word + '/'
        path = lead_path + numbered_name
        ordered_online_paths.append(path)

    
    new_online_path_trans_dict = {}
    for i, trans in enumerate(online_path_trans_dict.values()):
        new_online_path_trans_dict.update({ordered_online_paths[i]:trans})
    online_path_trans_dict = new_online_path_trans_dict


    online_val1 = online_path_trans_dict.keys()
    online_val1 = list(online_val1)[:900]
    online_test = online_path_trans_dict.keys()
    online_test = list(online_test)[900:2500]
    online_train = online_path_trans_dict.keys()
    online_train = list(online_train)[2500:]

    for key in online_val1:
        val1_gt.append((key, online_path_trans_dict[key]))
    for key in online_test:
        test_gt.append((key, online_path_trans_dict[key]))
    for key in online_train:
        train_gt.append((key, online_path_trans_dict[key]))

    # Deal with all the other names

    # Write the path and the transcriptions to the ground truth files for dataset generation.
    os.remove('train_gt.txt')
    os.remove('test_gt.txt')
    os.remove('val1_gt.txt')
    with open('train_gt.txt', 'w') as f:
        for path, trans in train_gt:
            f.write(path[2:] + '\t' + trans)
    with open('test_gt.txt', 'w') as f:
        for path, trans in test_gt:
            f.write(path[2:] + '\t' + trans)
    with open('val1_gt.txt', 'w') as f:
        for path, trans in val1_gt:
            f.write(path[2:] + '\t' + trans)


    # Convert offline images to greyscale (already done doesn't need to run anymore)
    # Convert the .tiff images to .png (also no longer needs to be done)
    # for online_path, offline_path, path0, grey_val, trans in path_grey_trans:
    #     if (online_path != ''):
    #         # os.rename(online_path, online_path + 'f')
    #         # online_path = online_path + 'f'
    #         online_im = cv2.imread(online_path)
    #         # print(online_path)
    #         # cv2.imshow('fig', online_im)
    #         # cv2.waitKey()
    #         # cv2.destroyAllWindows()
    #         cv2.imwrite(online_path[2:-4] + '.png', online_im)
        # offline_im = cv2.imread(offline_path)
        # offline_bin_im = binarize(offline_im, int(grey_val))

        # cv2.imwrite('../' + offline_path[2:], offline_bin_im)
        
    
        






if __name__ == '__main__':
    main()
#     '''
#     trainpaths = set_to_path(trainset)
#     testpaths_t = set_to_path(testset_t)
#     testpaths_v = set_to_path(testset_v)
#     testpaths_f = set_to_path(testset_f)
#     '''
#     # Some hacky shit because pandas sucks sometimes
#     line_list = []
#     with open('./IAM_ascii/lines.txt') as f:
#         line_list = f.readlines()
#     line_list = [[x.split(' ')[0], x.split(' ')[len(x.split(' '))-1]] for x in line_list if x[0] != '#']
#     line_list = [[x[0], x[1][:-1]] for x in line_list]
#     words = np.array(line_list)
#     words[:, 0] = word_to_path(words[:, 0])
#     path_pieces = words[:, 0]
#     path_pieces = [x.split('-') for x in path_pieces]
#     words[:, 0] = path_pieces_to_path(path_pieces)
    
#     '''
#     match_traintest = []
#     match_traintest = [x.split('/')[0] + '/' + x.split('/')[1] + '/'  for x in words[:, 0]]
    
#     trainpaths = np.array(trainpaths).flatten()
#     testpaths_t = np.array(testpaths_t).flatten()
#     testpaths_v = np.array(testpaths_v).flatten()
#     testpaths_f = np.array(testpaths_f).flatten()
#     match_traintest = np.array(match_traintest).flatten()
    
#     train_ids = getids(trainpaths, match_traintest)
#     testt_ids = getids(testpaths_t, match_traintest)
#     testv_ids = getids(testpaths_v, match_traintest)
#     testf_ids = getids(testpaths_f, match_traintest)
#     # We are not using any of these IDs now cuz they arent all there. :cry:
#     # Just split the entire words array into train/val and call it a day
#     '''   
#     with open('train_gt.txt', 'w') as f:
#         for item in words[:int(len(words)*.9)].tolist():
#             f.write(item[0] + '\t' + item[1] + '\n')
#     with open('val_gt.txt', 'w') as f:
#         for item in words[int(len(words)*.9):].tolist():
#             f.write(item[0] + '\t' + item[1] + '\n')
    
#     # Now print out the files

#     ''' with open('train_gt.txt', 'w') as f:
#         for item in words[train_ids].tolist():
#             f.write(item[0] + '\t' + item[1] + '\n')
#     with open('testt_gt.txt', 'w') as f:
#         for item in words[testt_ids].tolist():
#             f.write(item[0] + '\t' + item[1] + '\n')
#     with open('testv_gt.txt', 'w') as f:
#         for item in words[testv_ids].tolist():
#             f.write(item[0] + '\t' + item[1] + '\n')
#     with open('testf_gt.txt', 'w') as f:
#         for item in words[testf_ids].tolist():
#             f.write(item[0] + '\t' + item[1] + '\n')
#     '''
# def getids(paths, match_traintest):
#     idxs = [np.nonzero(np.where(match_traintest==x, 1, 0)) for x in paths]
#     idxs = np.array(idxs).flatten()
#     idxs = np.concatenate(idxs).ravel()
#     return idxs



# def path_pieces_to_path(path_pieces):
#     paths = [x[0] + '/' + x[0] + '-' + x[1] + '/' + x[0]+'-'+x[1]+'-'+x[2]+'-'+x[3] for x in path_pieces]
#     return paths

# def word_to_path(labels):
#     labels = [x + '.png' for x in labels]
#     return labels

# def set_to_path(df):
#     vals =list(df.to_numpy().flatten())
#     print(vals)
#     paths = [x.split('-')[0] + '/'  + x.split('-')[1] + '/' for x in vals]
#     paths = [x.split(' ')[1] for x in paths]
#     return np.array(paths)

