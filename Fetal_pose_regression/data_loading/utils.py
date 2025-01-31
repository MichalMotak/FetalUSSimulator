import os
import pandas as pd
import json
import numpy as np
import cv2

from numpy import amin, amax


from torch.utils.data.dataset import Subset



DATA_PATH = "/home/mm/projects/JNU-IFM/dataset/data"

LABEL_DICT = {3: "None", 4: "OnlySP", 5: "OnlyHead", 6: "SPHead"}

CASES_NAMES = os.listdir(DATA_PATH)


def normalize(img):
    # This function normalizes image to range [0,1]
    low_lim = amin(img)
    high_lim = amax(img)
    if(high_lim - low_lim) == 0.0:
        norm_img = (img-low_lim)/(high_lim - low_lim + 10e-9)
    else:
        norm_img = (img-low_lim)/(high_lim - low_lim)
    return norm_img

def cropp_to_square_get_margins(mask_argmax, org_shape):
    
    mask_cords_y = np.indices(org_shape[:2])[1] 
    mask_cords_y_1 = mask_cords_y[mask_argmax.astype(bool)]

    # =========
    if org_shape[1] > org_shape[0]:
        diff = org_shape[1] - org_shape[0]
        new_y_shape = org_shape[1] - diff
        assert(org_shape[0] == new_y_shape)
        y_limit0 = (diff//2)
        y_limit1 = org_shape[1] - (diff//2)

    # print(y_limit0, np.min(mask_cords_y_1), y_limit1, np.max(mask_cords_y_1))

    if y_limit0 > np.min(mask_cords_y_1):
        y_coords_on_left = y_limit0 - np.min(mask_cords_y_1)
    else:
        y_coords_on_left = None 

    if np.max(mask_cords_y_1) >= y_limit1:
        y_coords_on_right = np.max(mask_cords_y_1) - y_limit1
    else:
        y_coords_on_right = None


    n = 20
    # print(y_coords_on_left, y_coords_on_right)
    if y_coords_on_left and not y_coords_on_right: # if pixels are 
        diff1 = diff//2 - y_coords_on_left - n
        diff2 = diff//2 + y_coords_on_left + n

    elif y_coords_on_right and not y_coords_on_left:
        diff1 = diff//2 + y_coords_on_right + n
        diff2 = diff//2 - y_coords_on_right - n

    elif y_coords_on_right == 0:
        diff1 = diff//2 + y_coords_on_right + n
        diff2 = diff//2 - y_coords_on_right - n

    elif y_coords_on_left == 0:
        diff1 = diff//2 - y_coords_on_left - n
        diff2 = diff//2 + y_coords_on_left + n

    # print(diff1, diff2)
    return diff1, diff2

def cropp_to_square(image, mask, margins = []):
    #
    # (1026, 1295) to (1026, 1026)

    if len(margins) == 0:
        if mask.shape[1] > mask.shape[0]:
            diff1 = ((mask.shape[1] - mask.shape[0])//2)
            diff2 = diff1

    else:
        assert len(margins) == 2
        diff1, diff2 = margins


    return_margines = [diff1, diff2]
    mask_cropped = mask[:,diff1 : mask.shape[1]-diff2-1]
    image_cropped = image[:,diff1 : mask.shape[1]-diff2-1]
    # print("m" , mask_cropped.shape)
    assert(mask_cropped.shape[0] == mask_cropped.shape[1] == mask.shape[0])
    assert(image_cropped.shape[0] == image_cropped.shape[1] == mask.shape[0])

    _, counts_mask = np.unique(mask, return_counts=True)
    _, counts_mask_cropped = np.unique(mask_cropped, return_counts=True)

    # print(counts_mask, counts_mask_cropped)

    if not (counts_mask[-1] == counts_mask_cropped[-1]): # if cropping led to cut of masks
        mask_shape = mask.shape
        mask_argmax = np.argmax(mask, axis = -1)
        mask_argmax = np.where(mask_argmax>0,1,0)
        
        margins_values = cropp_to_square_get_margins(mask_argmax, mask_shape) # search for new 
        image_cropped, mask_cropped, _ = cropp_to_square(image, mask, margins = margins_values)
        _, counts_mask_cropped2 = np.unique(mask_cropped, return_counts=True)

        # print(counts_mask, counts_mask_cropped2)
        # print(image_cropped.shape, mask_cropped.shape[:2])

        # Check if shapes and number of pixels for classes are proper
        assert(image_cropped.shape == mask_cropped.shape[:2] == (mask_shape[0],mask_shape[0]))
        assert(counts_mask[-1] == counts_mask_cropped2[-1])

        return_margines = margins_values


    return image_cropped, mask_cropped, return_margines

def reconstruct_square_mask_to_original(mask_square, margins):

    if len(mask_square.shape) == 3:
        mask_reconstructed = np.pad(mask_square, ((0,0), (margins[0], margins[1]+1), (0,0)), "constant", constant_values=0)
    elif len(mask_square.shape) == 2:
        mask_reconstructed = np.pad(mask_square, ((0,0), (margins[0], margins[1]+1)), "constant", constant_values=0)

    
    return mask_reconstructed

def reconstruct_square_to_original(image_org, image_square, mask_square, margins):
    
    assert len(margins) == 2

    print(image_org.shape[0]+ sum(margins))


    image_org_left = image_org[:, :margins[0]]
    image_org_right = image_org[:, -margins[1]-1:]

    print(image_org_left.shape, image_org_right.shape)

    image_reconstructed = np.concatenate([image_org_left, image_square, image_org_right], axis = 1)

    assert image_org.shape == image_reconstructed.shape

    # mask_reconstructed = np.pad(mask_square, iaxis_pad_width = (margins[0], margins[1]), iaxis = 1)
    mask_reconstructed = reconstruct_square_mask_to_original(mask_square, margins)
    
    assert image_org.shape == mask_reconstructed.shape[:2]

    assert (image_org == image_reconstructed).all()

    return image_reconstructed, mask_reconstructed




def load_image_mask(image_name):

    case_name, frame_id = image_name.split("_")

    case_paths = get_paths_for_case_name(case_name)
    case_images_names = get_file_names_list(case_paths['image_path'])
    case_images_names = sort_file_names_list(case_images_names)

    images = load_images_for_case(case_name)
    masks = load_masks_for_case(case_name)

    ind = case_images_names.index(image_name+".png")
    return images[ind], masks[ind]




def load_images_for_case(case_name):
    case_paths = get_paths_for_case_name(case_name)
        
    case_images_names = get_file_names_list(case_paths['image_path'])
    case_images_names = sort_file_names_list(case_images_names)
    # print(len(case_images_names))
    # print(case_images_names[:5])

    images_list = []
    
    for i in range(len(case_images_names)):
        path_to_file = os.path.join(case_paths['image_path'], case_images_names[i])
        image = cv2.imread(path_to_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = normalize(image).astype(np.float32)

        images_list.append(image)

    images_array = np.array(images_list)
    # print(images_array.shape)

    return images_array

def load_masks_for_case(case_name):
    case_paths = get_paths_for_case_name(case_name)
        
    case_images_names = get_file_names_list(case_paths['mask_enhance_path'])
    case_images_names = sort_file_names_list(case_images_names)

    images_list = []
    
    for i in range(len(case_images_names)):
        path_to_file = os.path.join(case_paths['mask_enhance_path'], case_images_names[i])
        mask = cv2.imread(path_to_file)
        images_list.append(mask)

    images_array = np.array(images_list)
    print(images_array.shape)

    return images_array


def get_paths_for_case_name(case_name):
    if case_name in CASES_NAMES:
        case_path = os.path.join(DATA_PATH, case_name)

        image_path = os.path.join(case_path, "image")
        mask_path = os.path.join(case_path, "mask")
        mask_enhance = os.path.join(case_path, "mask_enhance")
        csv = os.path.join(case_path, "frame_label.csv")

    output = {"image_path": image_path,
                "mask_path": mask_path,
                "mask_enhance_path": mask_enhance,
                "csv_file": csv}
    return output


# def get_paths_for_case_name(data_path, cases_names, case_name):
#     if case_name in cases_names:
#         case_path = os.path.join(data_path, case_name)

#         image_path = os.path.join(case_path, "image")
#         mask_path = os.path.join(case_path, "mask")
#         mask_enhance = os.path.join(case_path, "mask_enhance")
#         csv = os.path.join(case_path, "frame_label.csv")

#     output = {"image_path": image_path,
#                 "mask_path": mask_path,
#                 "mask_enhance_path": mask_enhance,
#                 "csv_file": csv}
#     return output


def get_file_names_list(path, sorted = True):
    # For image/mask/mask_enhance
    l_ = os.listdir(path)
    if sorted:
        l_ = sort_file_names_list(l_)

    return l_

def sort_file_names_list(l_):
    l2 = l_
    l2.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    return l2

def get_frame_id_from_name(img_file_name):
    return int(img_file_name.split("_")[1].split(".")[0])



def get_label_list_for_case(csv_path):
    df = pd.read_csv(csv_path)
    return df["frame_label"].values.tolist()
    
def save_testing_filenames(dataloader_subset: Subset, path_to_save: str):

    images_paths_list = dataloader_subset.dataset.images_paths_list
    masks_paths_list = dataloader_subset.dataset.masks_paths_list
 
    indices = dataloader_subset.indices

    test_images_paths_list = [images_paths_list[i] for i in indices]
    test_masks_paths_list = [masks_paths_list[i] for i in indices]

    dict_to_save = {"images_paths_list": test_images_paths_list,
                    "masks_paths_list": test_masks_paths_list}

    json_dict = json.dumps(dict_to_save)

    with open(path_to_save, "w") as f:
        f.write(json_dict)


##########################


def get_case_names_for_label(label, add_data_path, label_dict):
    df_final = pd.read_csv(os.path.join(add_data_path, "info_final.csv"), index_col=[0])
    label_search_name  = "OnlySP"
    label_search = [k for k, v in label_dict.items() if v == label_search_name][0]
    df_final_one_label = df_final[df_final['frame_label'] == label_search]

    # print(df_final_one_label.shape)
    # print(df_final_one_label.head(5))

    case_names = df_final_one_label['case_name'].unique()
    return case_names