
import scipy.io
import numpy as np
import os
import config



OUTPUT_CLASSES = config.num_classes

TRAIN_FRAC = config.train_frac #Fraction of data to be used for training in a class
TEST_FRAC = config.test_frac


    
    
def create_dataset():
    # Load dataset
    DATA_PATH = os.path.join(os.getcwd(),"Data")
    input_mat = scipy.io.loadmat(os.path.join(DATA_PATH, config.dataset_file_name))[config.dataset_mat_name]
    target_mat = scipy.io.loadmat(os.path.join(DATA_PATH, config.dataset_gt_file_name))[config.dataset_gt_mat_name]


    # Define global variables
    HEIGHT = input_mat.shape[0]
    WIDTH = input_mat.shape[1]
    BAND = input_mat.shape[2]
    PATCH_SIZE = config.patch_size
    TRAIN_PATCH,TRAIN_LABELS,TEST_PATCH,TEST_LABELS = [],[],[],[]
    CLASSES = [] 



    # Scale the input between [0,1]
    input_mat = np.pad(input_mat, ((PATCH_SIZE//2, PATCH_SIZE//2), (PATCH_SIZE//2, PATCH_SIZE//2), (0, 0)), 'symmetric')
    input_mat = input_mat.astype(float)
    input_mat -= np.min(input_mat)
    input_mat /= np.max(input_mat)


    # Calculate the mean of each channel for normalization
    MEAN_ARRAY = np.ndarray(shape=(BAND,),dtype=float)
    for i in range(BAND):
        MEAN_ARRAY[i] = np.mean(input_mat[:,:,i])


    def Patch(height_index,width_index):
        """
        Returns a mean-normalized patch, the top left corner of which 
        is at (height_index, width_index)
        
        Inputs: 
        height_index - row index of the top left corner of the image patch
        width_index - column index of the top left corner of the image patch
    
        Outputs:
        mean_normalized_patch - mean normalized patch of size (PATCH_SIZE, PATCH_SIZE) 
        whose top left corner is at (height_index, width_index)
        """
        transpose_array = np.transpose(input_mat,(2,0,1))
        height_slice = slice(height_index, height_index+PATCH_SIZE)
        width_slice = slice(width_index, width_index+PATCH_SIZE)
        patch = transpose_array[:, height_slice, width_slice]
        mean_normalized_patch = []
        for i in range(patch.shape[0]):
            mean_normalized_patch.append(patch[i] - MEAN_ARRAY[i]) 
    
        return np.array(mean_normalized_patch)


    # Collect all available patches of each class from the given image
    for i in range(OUTPUT_CLASSES):
        CLASSES.append([])
    for i in range(HEIGHT):
        for j in range(WIDTH):
            curr_inp = Patch(i,j)
            curr_tar = target_mat[i, j]
            if(curr_tar!=0): #Ignore patches with unknown landcover type for the central pixel
                CLASSES[curr_tar-1].append(curr_inp)


        


    # Make a split for each class
    for c in range(OUTPUT_CLASSES): #for each class
        class_population = len(CLASSES[c])
        patches_of_current_class = CLASSES[c]
        np.random.shuffle(patches_of_current_class)
        
        train_patch_num = int(class_population * TRAIN_FRAC)
        test_patch_num = int(class_population * TEST_FRAC)

        TRAIN_PATCH.extend(patches_of_current_class[:train_patch_num])
        TRAIN_LABELS = np.append(TRAIN_LABELS, np.full(train_patch_num, c, dtype=int))
        TEST_PATCH.extend(patches_of_current_class[train_patch_num:train_patch_num+test_patch_num])
        TEST_LABELS = np.append(TEST_LABELS, np.full(test_patch_num, c, dtype=int))


    # Train data devided into 8 files
    train_population_seg = len(TRAIN_PATCH)//7
    for i in range(8):
        train_dict = {}
        if i < 6:
            start = i * train_population_seg
            end = (i+1) * train_population_seg
        elif i < 7:
            start = i * train_population_seg
            end = (i+1) * train_population_seg - 1
        else:
            start = i * train_population_seg - 1
            end = len(TRAIN_PATCH)
        file_name = config.dataset + '_Train_'+str(PATCH_SIZE)+'_'+str(i+1)+'.mat'
        train_dict["train_patch"] = TRAIN_PATCH[start:end]
        train_dict["train_labels"] = TRAIN_LABELS[start:end]
        scipy.io.savemat(os.path.join(DATA_PATH, file_name),train_dict)


    # Test data devided into 8 files
    test_population_seg = len(TEST_PATCH)//7
    for i in range(8):
        test_dict = {}
        if i < 6:
            start = i * test_population_seg
            end = (i+1) * test_population_seg
        elif i < 7:
            start = i * test_population_seg
            end = (i+1) * test_population_seg - 1
        else:
            start = i * test_population_seg - 1
            end = len(TEST_PATCH)
        file_name = config.dataset + '_Test_'+str(PATCH_SIZE)+'_'+str(i+1)+'.mat'
        test_dict["test_patch"] = TEST_PATCH[start:end]
        test_dict["test_labels"] = TEST_LABELS[start:end]
        scipy.io.savemat(os.path.join(DATA_PATH, file_name),test_dict)

