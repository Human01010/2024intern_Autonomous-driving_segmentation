import os
import numpy as np
from PIL import Image
import cv2 # OpenCV for contour finding
from tqdm import tqdm
from collections import namedtuple

#--------------------------------------------------------------------------------
# Definitions from Cityscapes scripts
# (helpers/labels.py)
#--------------------------------------------------------------------------------
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create category images
                    # Refer to the website for more details.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluation or not

    'color'       , # The color of this label
    ]
)

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

# Create a mapping from label ID to train ID
id_to_trainid = {label.id: label.trainId for label in labels}
id_to_trainid_map_arr = np.zeros(max(id_to_trainid.keys()) + 1, dtype=np.uint8)
for id_val, trainid_val in id_to_trainid.items():
    if id_val >= 0: # Ignore license plate with id -1 for array mapping
        id_to_trainid_map_arr[id_val] = trainid_val

# Number of classes we care about (0-18 for trainId)
NUM_CLASSES = 19 # Should match your cityscapes.yaml nc
MIN_CONTOUR_AREA = 10 # Minimum pixel area for a contour to be considered

def convert_mask_to_yolo_segmentation(mask_path, output_txt_path):
    """
    Converts a single Cityscapes _gtFine_labelIds.png mask to YOLO segmentation format.
    Each distinct object of a class will be a separate polygon.
    """
    try:
        labelid_mask_pil = Image.open(mask_path)
        labelid_mask_np = np.array(labelid_mask_pil)
        img_h, img_w = labelid_mask_np.shape[:2]

        # Convert label IDs to train IDs
        trainid_mask_np = id_to_trainid_map_arr[labelid_mask_np]

        yolo_lines = []

        for train_id in range(NUM_CLASSES): # Iterate through trainIds 0-18
            # Create a binary mask for the current train_id
            current_class_mask = (trainid_mask_np == train_id).astype(np.uint8)

            if np.sum(current_class_mask) == 0:
                continue # Class not present in this image

            # Find contours for each object of the current class
            # cv2.RETR_LIST retrieves all contours,
            # cv2.RETR_EXTERNAL retrieves only the extreme outer contours.
            # For segmentation, if objects of the same class are separate, RETR_EXTERNAL is fine.
            # If they can have holes or be nested and you want all parts, RETR_TREE or RETR_CCOMP might be needed,
            # but YOLO format usually expects simple outer polygons per instance/segment.
            # Let's use RETR_EXTERNAL as it's simpler and often sufficient.
            contours, hierarchy = cv2.findContours(current_class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 3 or cv2.contourArea(contour) < MIN_CONTOUR_AREA: # Must be a polygon with some area
                    continue

                # Squeeze contour to remove redundant dimension: (N, 1, 2) -> (N, 2)
                contour_squeezed = contour.squeeze(1)

                # Normalize coordinates and flatten
                normalized_coords = []
                for point in contour_squeezed:
                    norm_x = point[0] / img_w
                    norm_y = point[1] / img_h
                    normalized_coords.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

                if normalized_coords:
                    yolo_lines.append(f"{train_id} {' '.join(normalized_coords)}")

        if yolo_lines:
            with open(output_txt_path, 'w') as f:
                for line in yolo_lines:
                    f.write(line + "\n")
        # else:
            # print(f"No valid contours found for {mask_path}, so no .txt file created.")

    except Exception as e:
        print(f"Error processing {mask_path}: {e}")


def process_cityscapes_split(root_dir, output_dir, split_name):
    """
    Processes a split (train/val) of the Cityscapes dataset.
    """
    gt_fine_dir = os.path.join(root_dir, 'gtFine', split_name)
    output_labels_split_dir = os.path.join(output_dir, 'labels', split_name)
    os.makedirs(output_labels_split_dir, exist_ok=True)

    # Also create corresponding images directory for YOLO structure (though we only copy/link images later)
    output_images_split_dir = os.path.join(output_dir, 'images', split_name)
    os.makedirs(output_images_split_dir, exist_ok=True)


    mask_files = []
    for city_folder in os.listdir(gt_fine_dir):
        city_path = os.path.join(gt_fine_dir, city_folder)
        if os.path.isdir(city_path):
            for fname in os.listdir(city_path):
                if fname.endswith('_gtFine_labelIds.png'):
                    mask_files.append(os.path.join(city_path, fname))

    print(f"Found {len(mask_files)} mask files in {split_name} split.")

    for mask_path in tqdm(mask_files, desc=f"Processing {split_name}"):
        base_fname = os.path.basename(mask_path)
        # Output filename should match the corresponding image filename structure
        # e.g., frankfurt_000000_000294_gtFine_labelIds.png -> frankfurt_000000_000294_leftImg8bit.txt
        txt_fname = base_fname.replace('_gtFine_labelIds.png', '_leftImg8bit.txt')
        output_txt_path = os.path.join(output_labels_split_dir, txt_fname)
        convert_mask_to_yolo_segmentation(mask_path, output_txt_path)

    print(f"Finished processing {split_name}. YOLO labels saved to {output_labels_split_dir}")
    print(f"Remember to also place/link your images in {output_images_split_dir}")


if __name__ == '__main__':
    # --- Configuration ---
    cityscapes_root_dir = './cityscapes' # Path to your main Cityscapes directory
                                         # (should contain gtFine and leftImg8bit)
    yolo_dataset_output_dir = './cityscapes_yolo_seg' # Directory to save the YOLO formatted dataset
    # ---------------------

    if not os.path.isdir(os.path.join(cityscapes_root_dir, 'gtFine')):
        print(f"Error: Cityscapes 'gtFine' directory not found in {cityscapes_root_dir}")
        exit()
    if not os.path.isdir(os.path.join(cityscapes_root_dir, 'leftImg8bit')):
        print(f"Error: Cityscapes 'leftImg8bit' directory not found in {cityscapes_root_dir}")
        exit()

    os.makedirs(yolo_dataset_output_dir, exist_ok=True)

    # Process train and val splits
    process_cityscapes_split(cityscapes_root_dir, yolo_dataset_output_dir, 'train')
    process_cityscapes_split(cityscapes_root_dir, yolo_dataset_output_dir, 'val')
    # process_cityscapes_split(cityscapes_root_dir, yolo_dataset_output_dir, 'test') # If you have test gtFine

    print("\nConversion complete.")
    print(f"YOLO formatted dataset structure created in: {yolo_dataset_output_dir}")
    print("Next steps:")
    print(f"1. Copy or symlink your actual images from '{cityscapes_root_dir}/leftImg8bit/train' to '{yolo_dataset_output_dir}/images/train'")
    print(f"2. Copy or symlink your actual images from '{cityscapes_root_dir}/leftImg8bit/val' to '{yolo_dataset_output_dir}/images/val'")
    print("3. Create a data.yaml file pointing to these new directories, for example:")
    print(f"""
# cityscapes_yolo.yaml
path: {os.path.abspath(yolo_dataset_output_dir)}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
# test: images/test # test images (optional)

# Classes
nc: {NUM_CLASSES}  # number of classes
names: [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
  ]  # class names (ensure order matches trainId 0-18)
""")
