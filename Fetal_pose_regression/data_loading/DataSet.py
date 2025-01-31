
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import compress

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# from skimage import io
# from skimage.io import imread, imshow
# from skimage.transform import resize
# from skimage.color import rgb2gray, rgba2rgb
import cv2

from data_loading.utils import cropp_to_square

from .utils import get_paths_for_case_name, get_file_names_list, normalize

LABEL_DICT = {3: "None", 4: "OnlySP", 5: "OnlyHead", 6: "SPHead"}


def returnNotMatches(a, b):
    a = set(a)
    b = set(b)
    return [list(b - a), list(a - b)]


class SegmentationDataSet(Dataset):

    def __init__(self, parameters):
        """

        Parameters:
            parameters (dict)
              Keys:
                data_path (str): path for data
                size (tuple):  single image shape (224,224)
                range (int, optional): range to limit number of images for dataloader

        """

        self.data_path = parameters["data_path"]
        self.size = parameters["size"]
        self.range = parameters["range"]
        self.selected_labels = parameters['selected_labels']

        self.to_transform = False


        # print(self.selected_labels , type(self.selected_labels))
        if type(self.selected_labels) == list:
            assert all(i in list(LABEL_DICT.values()) for i in self.selected_labels) or self.selected_labels[0] in ["all", "not_empty"]
        if len(self.selected_labels)==1:
            self.selected_labels = self.selected_labels[0]

        if self.selected_labels == "not_empty":
            self.selected_labels = ["OnlySP", "OnlyHead", "SPHead"]

        if self.selected_labels == "all":
            self.selected_labels = ["None", "OnlySP", "OnlyHead", "SPHead"]

        print("SELECTED LABELS: ", self.selected_labels)
        print("SIZE ", self.size)

        # paths to cases folders
        self.cases_paths_list = sorted([os.path.join(self.data_path, case_name) for case_name in sorted(os.listdir(self.data_path))])
        # paths to images/mask_enhance folders for cases 
        self.cases_images_paths_list = [os.path.join(x, "image") for x in self.cases_paths_list]
        self.cases_masks_paths_list = [os.path.join(x, "mask_enhance") for x in self.cases_paths_list]

        self.images_names_list = [get_file_names_list(case_path, sorted = True) for case_path in self.cases_images_paths_list]
        self.masks_names_list = [get_file_names_list(case_path, sorted = True) for case_path in self.cases_masks_paths_list]
        assert len(self.images_names_list) == len(self.masks_names_list)
        # Add full path to images_list, masks_list

        self.images_paths_list = []
        for case_path, case_images_names_list in zip(self.cases_images_paths_list, self.images_names_list):
            self.images_paths_list.extend([os.path.join(case_path, image_name) for image_name in case_images_names_list])

        self.masks_paths_list = []
        for case_path, case_masks_names_list in zip(self.cases_masks_paths_list, self.masks_names_list):
            self.masks_paths_list.extend([os.path.join(case_path, mask_name) for mask_name in case_masks_names_list])

        self.images_names_list = [i for x in self.images_names_list for i in x]
        self.masks_names_list = [i for x in self.masks_names_list for i in x]

        if self.selected_labels != "all":
            filtering_list_bool = self.get_selected_labels_filter_list()
            
            # print(len(filtering_list_bool), len(self.images_paths_list), len(self.masks_paths_list))

            # self.images_names_list_all = [i for x in self.images_names_list for i in x]
            # self.masks_names_list_all = [i for x in self.masks_names_list for i in x]

            images_names_filtered = list(compress(self.images_names_list, filtering_list_bool))
            masks_names_filtered = list(compress(self.masks_names_list, filtering_list_bool))

            images_paths_filtered = list(compress(self.images_paths_list, filtering_list_bool))
            masks_paths_filtered = list(compress(self.masks_paths_list, filtering_list_bool))

            self.images_paths_list = images_paths_filtered
            self.masks_paths_list = masks_paths_filtered

                # print(len(images_names_filtered), len(masks_names_filtered), len(images_paths_filtered), len(masks_paths_filtered))
                # assert (len(images_names_filtered) == len(masks_names_filtered) == len(images_paths_filtered) == len(masks_paths_filtered))
        
        # assert len(self.images_names_list) == len(self.masks_names_list)
        

        # print(len(self.cases_paths_list), len(self.images_names_list), len(self.masks_names_list))
        print(len(self.images_paths_list), len(self.masks_paths_list))

        if (self.range > 0) and (self.range <= len(self.images_paths_list)):
            self.images_paths_list = self.images_paths_list[:self.range]
            self.masks_paths_list = self.masks_paths_list[:self.range]
            self.images_names_list = self.images_names_list[:self.range]
            self.masks_names_list = self.masks_names_list[:self.range]
        #     print(len(self.cases_paths_list), len(self.images_names_list), len(self.masks_names_list))
        #     print(len(self.images_paths_list), len(self.masks_paths_list))

        # print(len(self.images_paths_list), self.images_paths_list[:5])
        # print(len(self.masks_paths_list), self.masks_paths_list[:5])

        # print(len(self.images_names_list), self.images_names_list[:5])
        # print(len(self.masks_names_list), self.masks_names_list[:5])

        assert(len(self.images_names_list) == len(self.masks_names_list))
        assert(len(self.images_paths_list) == len(self.masks_paths_list))

        # print(self.masks_names_list[:3])
        #Remove "_mask" to self.masks_list elements, '20190830T115515_9_mask.png' - >  '20190830T115515_9.png'
        masks_names_list2 = [i.replace('_mask','') for i in self.masks_names_list]
        if self.images_names_list != masks_names_list2:
            n = returnNotMatches(self.images_names_list, masks_names_list2)
            raise Exception("Images and masks lists don't match: ", n)



    def get_selected_labels_filter_list(self):

        filtering_list_bool = []

        self.cases_names = sorted(os.listdir(self.data_path))
        # cases_paths_list = sorted([os.path.join(self.data_path, case_name) for case_name in self.cases_names])

        for case_name in self.cases_names:
            case_paths = get_paths_for_case_name(case_name)
            df = pd.read_csv(case_paths['csv_file'], index_col=[0])
            df.sort_values("frame_id", inplace=True)
            label_list = df["frame_label"].values.tolist()

            label_values = [k for k, v in LABEL_DICT.items() if v in self.selected_labels]


            label_list_filtered_bool = [True if x in label_values else False for x in label_list]
            # print(label_list[:5])
            # print(label_list_filtered_bool[:5])

            # print(list(compress(label_list, label_list_filtered_bool)))

            # print(len(label_list_filtered_bool), label_list_filtered_bool[:5])
            filtering_list_bool.extend(label_list_filtered_bool)

        return filtering_list_bool


    def __len__(self):
        """ __len__ function """
        if len(self.images_paths_list) == len(self.masks_paths_list):
            return len(self.images_paths_list)

    def get_by_index(self, idx):
        return self.__getitem__(idx)


    def __getitem__(self, index:int=0):
        """ Load images and masks from their lists by given index

        Arguments:
          index (int) : index to get one pair of image-mask
        Returns:
          image (numpy.ndarray): output image
          mask (numpy.ndarray): image mask
        """

        image_path = self.images_paths_list[index]
        mask_path = self.masks_paths_list[index]
        image, mask = self.load_preprocess_image_mask(image_path, mask_path)
        # print(image.shape, mask.shape)

        return image, mask
    
    def load_preprocess_image_mask(self, image_path, mask_path):
        """ Load and preprocess image and mask 
        Arguments:
          image_path (str): path to image
          mask_path (str): path to mask 

        Returns:
          image_th (torch.Tensor): output image, (B,1,W,H)
          mask_th (torch.Tensor): image mask, (B,W,H,1)
        """

        # load image and mask in itk format
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(mask_path)

        image, mask, _ = cropp_to_square(image, mask)
        image = cv2.resize(image, self.size, interpolation = cv2.INTER_NEAREST)
        mask = cv2.resize(mask, self.size, interpolation = cv2.INTER_NEAREST)        
        image = normalize(image).astype(np.float32)

        mask_th = torch.from_numpy(mask)
        image_th = torch.from_numpy(image).unsqueeze(0)

        # 0 - Background, 1 - Head, 2 - SP
        mask_th = torch.argmax(mask_th, dim=-1).unsqueeze(-1)

        return image_th, mask_th


    def visualize_by_index(self, index, as_gray = True):
        """ Visualize image and mask in plt.figure based on index"""
        
        image, mask = self.__getitem__(index)
        image = image[0].numpy()
        mask = mask.numpy()
        # mask = mask[0]
        print(image.shape, mask.shape)
        print(np.max(image), np.min(image))
        print(np.unique(mask))
      
        plt.figure(figsize = (8,3))
        plt.subplot(121)
        if as_gray:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
     
        plt.axis("off")
        plt.subplot(122)
        if as_gray:
            plt.imshow(mask, cmap="gray")
        else:
            plt.imshow(mask)
        plt.axis("off")
        plt.show()


    def visualize_random_indexes(self, number_of_images):
        indexes_list = np.random.randint(0,len(self.images_paths_list), size = number_of_images)
        print(indexes_list)
        for index in indexes_list:
            self.visualize_by_index(index)


























class SegmentationDataSetCase(Dataset):

    def __init__(self, parameters):
        """

        Parameters:
            parameters (dict)
              Keys:
                images_path (str): path for images
                masks_path (str): path for masks
                color_mode (str, "gray" or "rgb"): color mode type
                size (tuple):  single image shape (224,224)
                range (int, optional): range to limit number of images for dataloader
                batch_size (int): batch size for generator

        """

        self.data_path = parameters["data_path"]
        self.size = parameters["size"]
        self.range = parameters["range"]
        self.selected_labels = parameters['selected_labels']

        print(self.selected_labels , type(self.selected_labels))
        if type(self.selected_labels) == list:
            assert all(i in list(LABEL_DICT.values()) for i in self.selected_labels)
        else:
            assert self.selected_labels == "all"

        # paths to cases folders
        self.cases_paths_list = sorted([os.path.join(self.data_path, case_name) for case_name in sorted(os.listdir(self.data_path))])
        # paths to images/mask_enhance folders for cases 
        self.cases_images_paths_list = [os.path.join(x, "image") for x in self.cases_paths_list]
        self.cases_masks_paths_list = [os.path.join(x, "mask_enhance") for x in self.cases_paths_list]

        self.images_names_list = [get_file_names_list(case_path, sorted = True) for case_path in self.cases_images_paths_list]
        self.masks_names_list = [get_file_names_list(case_path, sorted = True) for case_path in self.cases_masks_paths_list]
        assert len(self.images_names_list) == len(self.masks_names_list)

        # Add full path to images_list, masks_list
        self.images_paths_list = []
        for case_path, case_images_names_list in zip(self.cases_images_paths_list, self.images_names_list):
            self.images_paths_list.extend([os.path.join(case_path, image_name) for image_name in case_images_names_list])

        self.masks_paths_list = []
        for case_path, case_masks_names_list in zip(self.cases_masks_paths_list, self.masks_names_list):
            self.masks_paths_list.extend([os.path.join(case_path, mask_name) for mask_name in case_masks_names_list])

        self.images_names_list = [i for x in self.images_names_list for i in x]
        self.masks_names_list = [i for x in self.masks_names_list for i in x]

        if self.selected_labels != "all":
            filtering_list_bool = self.get_selected_labels_filter_list()
            
            # print(len(filtering_list_bool), len(self.images_paths_list), len(self.masks_paths_list))

            # self.images_names_list_all = [i for x in self.images_names_list for i in x]
            # self.masks_names_list_all = [i for x in self.masks_names_list for i in x]

            images_names_filtered = list(compress(self.images_names_list, filtering_list_bool))
            masks_names_filtered = list(compress(self.masks_names_list, filtering_list_bool))

            images_paths_filtered = list(compress(self.images_paths_list, filtering_list_bool))
            masks_paths_filtered = list(compress(self.masks_paths_list, filtering_list_bool))

            self.images_paths_list = images_paths_filtered
            self.masks_paths_list = masks_paths_filtered


        print(len(self.cases_paths_list), len(self.images_names_list), len(self.masks_names_list))
        print(len(self.images_paths_list), len(self.masks_paths_list))

        if (self.range > 0) and (self.range <= len(self.images_paths_list)):
            self.images_paths_list = self.images_paths_list[:self.range]
            self.masks_paths_list = self.masks_paths_list[:self.range]
            self.images_names_list = self.images_names_list[:self.range]
            self.masks_names_list = self.masks_names_list[:self.range]
            print(len(self.cases_paths_list), len(self.images_names_list), len(self.masks_names_list))
            print(len(self.images_paths_list), len(self.masks_paths_list))

        print(len(self.images_paths_list), self.images_paths_list[:5])
        print(len(self.masks_paths_list), self.masks_paths_list[:5])

        # print(len(self.images_names_list), self.images_names_list[:5])
        # print(len(self.masks_names_list), self.masks_names_list[:5])

        assert(len(self.images_names_list) == len(self.masks_names_list))
        assert(len(self.images_paths_list) == len(self.masks_paths_list))

        # print(self.masks_names_list[:3])
        #Remove "_mask" to self.masks_list elements, '20190830T115515_9_mask.png' - >  '20190830T115515_9.png'
        masks_names_list2 = [i.replace('_mask','') for i in self.masks_names_list]
        if self.images_names_list != masks_names_list2:
            n = returnNotMatches(self.images_names_list, masks_names_list2)
            raise Exception("Images and masks lists don't match: ", n)


    def get_selected_labels_filter_list(self):

        filtering_list_bool = []


        self.cases_names = sorted(os.listdir(self.data_path))
        # cases_paths_list = sorted([os.path.join(self.data_path, case_name) for case_name in self.cases_names])

        for case_name in self.cases_names:
            case_paths = get_paths_for_case_name(self.data_path, self.cases_names, case_name)
            df = pd.read_csv(case_paths['csv_file'], index_col=[0])
            df.sort_values("frame_id", inplace=True)
            label_list = df["frame_label"].values.tolist()

            label_values = [k for k, v in LABEL_DICT.items() if v in self.selected_labels]


            label_list_filtered_bool = [True if x in label_values else False for x in label_list]
            # print(label_list[:5])
            # print(label_list_filtered_bool[:5])

            # print(list(compress(label_list, label_list_filtered_bool)))

            # print(len(label_list_filtered_bool), label_list_filtered_bool[:5])
            filtering_list_bool.extend(label_list_filtered_bool)

        return filtering_list_bool


    def __len__(self):
        """ __len__ function """
        if len(self.images_paths_list) == len(self.masks_paths_list):
            return len(self.images_paths_list)

    def get_by_index(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, index:int=0):
        """ Load images and masks from their lists by given index

        Arguments:
          index (int) : index to get one pair of image-mask
        Returns:
          image (numpy.ndarray): output image
          mask (numpy.ndarray): image mask
        """

        image_path = self.images_paths_list[index]
        mask_path = self.masks_paths_list[index]
        print(image_path, mask_path)
        image, mask = self.load_preprocess_image_mask(image_path, mask_path)
                    
        return image, mask
    
    def load_preprocess_image_mask(self, image_path, mask_path):
        """ Load and preprocess image and mask 
        Arguments:
          image_path (str): path to image
          mask_path (str): path to mask 

        Returns:
          image (numpy.ndarray): output image
          mask (numpy.ndarray): image mask
        """

        # load image and mask in itk format
        image = imread(image_path)
        mask = imread(mask_path)

        # print(np.max(image), np.min(image))
        # print(np.unique(mask))

        image_th = torch.from_numpy(image)

        # resize_transform = transforms.Resize(size=self.size)
        # image = resize_transform(image)
        # mask = resize_transform(mask)  


        transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(self.size),
                    transforms.PILToTensor()])

        image = transform(image_th)
        image = image.numpy()
        image = normalize(image).astype(np.float32)


        mask = mask.transpose(2, 0, 1)
        mask_th = torch.from_numpy(mask)
        # transform = transforms.Compose([
        #             transforms.ToPILImage(),
        #             transforms.Resize(self.size),
        #             transforms.PILToTensor()])
        # print(mask_th.shape)
        mask2 = transform(mask_th)
        mask2 = mask2.permute(1, 2, 0)/255
        mask2[:,:,2] = (mask2[:,:,2] > 0.5)
        mask2[:,:,1] = (mask2[:,:,1] > 0.5)
        mask2[:,:,0] = (mask2[:,:,0] > 0.5)

        image_out = torch.from_numpy(image)

        return image_out, mask2

    def visualize_by_index(self, index, as_gray = True):
        """ Visualize image and mask in plt.figure based on index"""
        
        image, mask = self.__getitem__(index)
        image = image[0].numpy()
        mask = mask.numpy()
        # mask = mask[0]
        print(image.shape, mask.shape)
        print(np.max(image), np.min(image))
        print(np.unique(mask))
      
        plt.figure(figsize = (8,3))
        plt.subplot(121)
        if as_gray:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
     
        plt.axis("off")
        plt.subplot(122)
        if as_gray:
            plt.imshow(mask, cmap="gray")
        else:
            plt.imshow(mask)
        plt.axis("off")
        plt.show()


    def visualize_random_indexes(self, number_of_images):
        indexes_list = np.random.randint(0,len(self.images_paths_list), size = number_of_images)
        print(indexes_list)
        for index in indexes_list:
            self.visualize_by_index(index)























class CaseDataLoader():

    def __init__(self, data_path, cases_names, case_name):

        self.data_path = data_path
        self.cases_names = cases_names

        self.case_name = case_name
        self.paths = get_paths_for_case_name(self.data_path, self.cases_names, self.case_name)
        self.images_names = get_file_names_list(self.paths['image_path'], sorted = True)
        self.masks_enh_names = get_file_names_list(self.paths['mask_enhance_path'], sorted = True)

        # print(self.masks_enh_names)
        self.images_paths = [os.path.join(self.paths['image_path'], i) for i in self.images_names]
        self.masks_enh_paths = [os.path.join(self.paths['mask_enhance_path'], i) for i in self.masks_enh_names]

        # Setup variables

        self.df = pd.read_csv(self.paths['csv_file'], index_col=[0])
        self.label_list = self.df["frame_label"].values.tolist()
        # print(self.label_list[:5])

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_df):
        self._df = new_df 

    @property
    def label_list(self):
        return self._label_list

    @label_list.setter
    def label_list(self, new_label_list):
        self._label_list = new_label_list 

    def get_label_for_frame(self, frame_number):

        if frame_number in self.df['frame_id'].values.tolist():
            return self.df[self.df['frame_id'] == frame_number]['frame_label'].values[0]
        else:
            print(f"Frame number {frame_number} not in data")

    def load_data(self, type_, paths_tuple):

        images_list = []
        
        for name, path in zip(*paths_tuple):
            img = plt.imread(path)
            # Preprocessing
            if type_ == "images":
                img_processed = self.preprocess_image(img)
            else:
                img_processed = self.preprocess_mask(img)

            images_list.append(img_processed)

        images_array = np.array(images_list)
        return images_array

    def preprocess_image(self, image):
        return image

    def preprocess_mask(self, mask):
        return mask


    def load_data_all(self, type_ ):

        assert type_ in ("images", "masks")

        if type_ == "images":
            paths_tuple = (self.images_names, self.images_paths)
        else:
            paths_tuple = (self.masks_enh_names, self.masks_enh_paths)
        
        images_array = self.load_data(type_, paths_tuple)
        return images_array

    def load_data_filtered(self, params, return_add_info = False):
        # print(LABEL_DICT)

        assert params['type'] in ("images", "masks")
        assert params['mode'] in ("only", "exclude", "multiple")
        
        type_ = params['type']
        mode_ = params['mode']

        if mode_ == "multiple":
            assert type(params['values']) == list
            assert all(i in list(LABEL_DICT.values()) for i in params['values'])

        else:
            assert type(params['values']) == str
            assert params['values'] in list(LABEL_DICT.values())
        
        values_search_ = params['values']

        filtering_list_bool = self.get_filtering_list(mode_, values_search_)

        label_list_filtered = list(compress(self.label_list, filtering_list_bool))
        images_names_filtered = list(compress(self.images_names, filtering_list_bool))
        masks_enh_names_filtered = list(compress(self.masks_enh_names, filtering_list_bool))

        images_paths_filtered = list(compress(self.images_paths, filtering_list_bool))
        masks_enh_paths_filtered = list(compress(self.masks_enh_paths, filtering_list_bool))

        # print(len(label_list_filtered), label_list_filtered[:5])
        # print(len(images_paths_filtered), images_paths_filtered[:5])
        # print(len(masks_enh_paths_filtered), masks_enh_paths_filtered[:5])

        if type_ == "images":
            paths_tuple = (images_names_filtered, images_paths_filtered)
            # add_info = (label_list_filtered, images_names_filtered)
        else:
            paths_tuple = (masks_enh_names_filtered, masks_enh_paths_filtered)
            # add_info = (label_list_filtered, masks_enh_names_filtered)

        images_array = self.load_data(type_, paths_tuple)

        if return_add_info:
            return images_array, paths_tuple

        else:
            return images_array


    def get_filtering_list(self, mode_, values_search):

        if mode_ == "only":
            label_unique_names = [LABEL_DICT[i] for i in list(set(self.label_list))]
            if values_search in label_unique_names:

                label_value = [k for k, v in LABEL_DICT.items() if v == values_search][0]
                # print("label_value ", label_value)
                label_list_filtered_bool = [True if x == label_value else False for x in self.label_list]

            else:
                print(f"Searched variable '{values_search}' not in unique elements {label_unique_names}")
                return 0

        if mode_ == "exclude":
            label_values = [k for k, v in LABEL_DICT.items() if v != values_search]
            # print("label_values ", label_values)
            label_list_filtered_bool = [True if x in label_values else False for x in self.label_list]

        if mode_ == "multiple":
            label_values = [k for k, v in LABEL_DICT.items() if v in values_search]
            # print("label_values ", label_values)

            label_list_filtered_bool = [True if x in label_values else False for x in self.label_list]

        return label_list_filtered_bool

