from __future__ import print_function
import config
from config import *

# Create directory structure
def create_dataset_split_structure():
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    if not os.path.exists(dataset_train_path):
        os.makedirs(dataset_train_path)

    if not os.path.exists(dataset_val_path):
        os.makedirs(dataset_val_path)

    if not os.path.exists(dataset_test_path):
        os.makedirs(dataset_test_path)
        
        
# Get category names list
def get_category_names():
    category_names = []
    with open('./anno/list_category_cloth.txt') as file_list_category_cloth:
        next(file_list_category_cloth)
        next(file_list_category_cloth)
        for line in file_list_category_cloth:
            word=line.strip()[:-1].strip().replace(' ', '_')
            category_names.append(word)
    return category_names



# Create category dir structure
def create_category_structure(category_names):
    for idx,category_name in enumerate(category_names):
         if idx < 50:           
                #Train
                category_path_name=os.path.join(dataset_train_path, category_name)
                logging.debug('category_path_name {}'.format(category_path_name))
                if not os.path.exists(os.path.join(category_path_name)):
                    os.makedirs(category_path_name)
                
                
                # Validation
                category_path_name=os.path.join(dataset_val_path, category_name)
                logging.debug('category_path_name {}'.format(category_path_name))
                if not os.path.exists(os.path.join(category_path_name)):
                    os.makedirs(category_path_name)
                
                
                # Test
                category_path_name=os.path.join(dataset_test_path, category_name)
                logging.debug('category_path_name {}'.format(category_path_name))
                if not os.path.exists(os.path.join(category_path_name)):
                     os.makedirs(category_path_name)
               
               
        
def get_dataset_split_name(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            dataset_split_name=line.split()[1]
            logging.debug('dataset_split_name {}'.format(dataset_split_name))
            return dataset_split_name.strip()
        
        
        
        
def get_gt_bbox(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            x1=int(line.split()[1])
            y1=int(line.split()[2])
            x2=int(line.split()[3])
            y2=int(line.split()[4])
            bbox = [x1, y1, x2, y2]
            logging.debug('bbox {}'.format(bbox))
            return bbox



def display_bbox(image_path_name, boxA, boxB):
    logging.debug('image_path_name {}'.format(image_path_name))

    # load image
    img = skimage.io.imread(image_path_name)
    logging.debug('img {}'.format(type(img)))

    # Draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)

    x, y, w, h = boxA[0], boxA[1], boxA[2]-boxA[0], boxA[3]-boxA[1]
    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=1)
    ax.add_patch(rect)
    logging.debug('GT: boxA {}'.format(boxA))
    logging.debug('   x    y    w    h')
    logging.debug('{:4d} {:4d} {:4d} {:4d}'.format(x, y, w, h))

    x, y, w, h = boxB[0], boxB[1], boxB[2]-boxB[0], boxB[3]-boxB[1]
    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
    logging.debug('boxB {}'.format(boxB))
    logging.debug('   x    y    w    h')
    logging.debug('{:4d} {:4d} {:4d} {:4d}'.format(x, y, w, h))

    plt.show()
    
    
    
    
    
def calculate_bbox_crop_and_save_img(image_path_name, dataset_image_path, gt_x1, gt_y1, gt_x2, gt_y2):

    logging.debug('dataset_image_path {}'.format(dataset_image_path))
    logging.debug('image_path_name {}'.format(image_path_name))

    image_name = image_path_name.split('/')[-1].split('.')[0]
    logging.debug('image_name {}'.format(image_name))

    img_read = Image.open(image_path_name)
    logging.debug('{} {} {}'.format(img_read.format, img_read.size, img_read.mode))

    # Ground Truth
    image_save_name = image_path_name.split('/')[-2] + '_' + image_path_name.split('/')[-1].split('.')[0]
    image_save_path_name = dataset_image_path + '/' + image_save_name + '_gt_' +  str(gt_x1) + '-' + str(gt_y1) + '-' + str(gt_x2) + '-' + str(gt_y2) + '.jpg'
    logging.debug('image_save_path_name {}'.format(image_save_path_name))
    #img_crop = img_read.crop((gt_y1, gt_x1, gt_y2, gt_x2))
    img_crop = img_read.crop((gt_x1, gt_y1, gt_x2, gt_y2))
    img_crop.save(image_save_path_name)
    logging.debug('img_crop {} {} {}'.format(img_crop.format, img_crop.size, img_crop.mode))
    
    
    
    
category_name_generate = ['Anorak', 'Blazer', 'Blouse', 'Bomber', 'Button-Down', 'Cardigan', 'Flannel', 'Halter', 'Henley', 'Hoodie', 'Jacket', 'Jersey', 'Parka', 'Peacoat', 'Poncho', 'Sweater', 'Tank', 'Tee', 'Top', 'Turtleneck', 'Capris', 'Chinos', 'Culottes', 'Cutoffs', 'Gauchos', 'Jeans', 'Jeggings', 'Jodhpurs', 'Joggers', 'Leggings', 'Sarong', 'Shorts', 'Skirt', 'Sweatpants', 'Sweatshorts', 'Trunks', 'Caftan', 'Cape', 'Coat', 'Coverup', 'Dress', 'Jumpsuit', 'Kaftan', 'Kimono', 'Nightdress', 'Onesie', 'Robe', 'Romper', 'Shirtdress', 'Sundress']

def generate_dataset_images(category_names):

    max_categories = 50
    count=0
    with open('./anno/list_bbox.txt') as file_list_bbox_ptr:
        with open('./anno/list_category_img.txt') as file_list_category_img:
            with open('./Eval/list_eval_partition.txt', 'r') as file_list_eval_ptr:

                next(file_list_category_img)
                next(file_list_category_img)
                idx_crop=1
                for line in file_list_category_img:
                    line = line.split()
                    image_path_name = line[0]
                    logging.debug('image_path_name {}'.format(image_path_name))                                 # img/Tailored_Woven_Blazer/img_00000051.jpg
                    image_name = line[0].split('/')[-1]
                    logging.debug('image_name {}'.format(image_name))                                           # image_name img_00000051.jpg
                    image_full_name = line[0].replace('/', '_')
                    logging.debug('image_full_name {}'.format(image_full_name))                                 # img_Tailored_Woven_Blazer_img_00000051.jpg
                    image_category_index=int(line[1:][0]) - 1
                    logging.debug('image_category_index {}'.format(image_category_index))                       # 2

                    if category_names[image_category_index] not in category_name_generate:
                        logging.debug('Skipping {} {}'.format(category_names[image_category_index], image_path_name))
                        continue


                    if image_category_index < max_categories:

                        dataset_image_path = ''
                        dataset_split_name = get_dataset_split_name(image_path_name, file_list_eval_ptr)

                        if dataset_split_name == "train":
                            dataset_image_path = os.path.join(dataset_train_path, category_names[image_category_index])
                        elif dataset_split_name == "val":
                            dataset_image_path = os.path.join(dataset_val_path, category_names[image_category_index])
                        elif dataset_split_name == "test":
                            dataset_image_path = os.path.join(dataset_test_path, category_names[image_category_index])
                        else:
                            logging.error('Unknown dataset_split_name {}'.format(dataset_image_path))
                            exit(1)

                        logging.debug('image_category_index {}'.format(image_category_index))
                        logging.debug('category_names {}'.format(category_names[image_category_index]))
                        logging.debug('dataset_image_path {}'.format(dataset_image_path))

                        # Get ground-truth bounding boxes
                        gt_x1, gt_y1, gt_x2, gt_y2 = get_gt_bbox(image_path_name, file_list_bbox_ptr)                              # Origin is top left, x1 is distance from y axis;
                                                                                                                                   # x1,y1: top left coordinate of crop; x2,y2: bottom right coordinate of crop
                        logging.debug('Ground bbox:  gt_x1:{} gt_y1:{} gt_x2:{} gt_y2:{}'.format(gt_x1, gt_y1, gt_x2, gt_y2))

                        image_path_name_src = os.path.join('.', image_path_name)
                        logging.debug('image_path_name_src {}'.format(image_path_name_src))

                        calculate_bbox_crop_and_save_img(image_path_name_src, dataset_image_path, gt_x1, gt_y1, gt_x2, gt_y2)

                    
                        idx_crop = idx_crop + 1
                        logging.debug('idx_crop {}'.format(idx_crop))

                 
                    count = count+1
                    logging.info('count {} {}'.format(count, dataset_image_path))

    
    
 
def display_category_data():
    for path in [dataset_train_path, dataset_val_path, dataset_test_path]:
        logging.info('path {}'.format(path))
        path1, dirs1, files1 = next(os.walk(path))
        file_count1 = len(files1)
        for dirs1_name in dirs1:
            path2, dirs2, files2 = next(os.walk(os.path.join(path, dirs1_name)))
            file_count2 = len(dirs2)
            logging.info('{:20s} : {}'.format(dirs1_name, file_count2))
            
    


if __name__ == '__main__':

    create_dataset_split_structure()
    category_names = get_category_names()
    logging.debug('category_names {}'.format(category_names))
    create_category_structure(category_names)
    generate_dataset_images(category_names)
    display_category_data()

