patch_size = 7
dataset = "IN"  # "IN", "UP" or "Salinas"

if(dataset == "IN"):
    dataset_file_name = "Indian_pines_corrected.mat"
    dataset_mat_name = "indian_pines_corrected"
    dataset_gt_file_name = "Indian_pines_gt.mat"
    dataset_gt_mat_name = "indian_pines_gt"
    num_classes = 16
    channels = 200
    train_frac = 0.15
    test_frac = 0.75
    contextual_kernel_num = 128
    ssdc_kernel_num = 48
elif(dataset == "UP"):
    dataset_file_name = "PaviaU.mat"
    dataset_mat_name = "paviaU"
    dataset_gt_file_name = "PaviaU_gt.mat"
    dataset_gt_mat_name = "paviaU_gt"
    num_classes = 9
    channels = 103
    train_frac = 0.05
    test_frac = 0.80
    contextual_kernel_num = 128
    ssdc_kernel_num = 32
elif(dataset == "Salinas"):
    dataset_file_name = "Salinas_corrected.mat"
    dataset_mat_name = "salinas_corrected"
    dataset_gt_file_name = "Salinas_gt.mat"
    dataset_gt_mat_name = "salinas_gt"
    num_classes = 16
    channels = 204
    train_frac = 0.05
    test_frac = 0.80
    contextual_kernel_num = 192
    ssdc_kernel_num = 16
else:
    assert False, "Dataset is not available."
