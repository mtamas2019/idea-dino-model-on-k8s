from pycocotools.coco import COCO
import os, logging, zipfile, shutil, sys, requests, wget, json

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)

# Env variables
dataset_dir = os.environ.get('DATASET_DIR', 'COCODIR')
download_percent = os.environ.get('DATASET_DOWNLOAD_SIZE_PERCENT', '100')
clean_dataset_dir = os.environ.get('CLEAN_DATASET_DIR', 'Yes')
recreate_annotations = os.environ.get('RECREATE_ANNONATIONS', 'Yes')

try:
   download_percent = int(download_percent)
except Exception as e:
    logging.error('DATASET_DOWNLOAD_SIZE_PERCENT need to be integer!')
    sys.exit(1)

# Functions
def clear_directory(target_dir):
  "Clears out a directory by removing all files inside"
  if os.path.exists(target_dir) and os.path.isdir(target_dir):
    for filename in os.listdir(target_dir):
      file_path = os.path.join(target_dir, filename)
      try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
          os.unlink(file_path)
      except Exception as e:
          logging.error('Failed to delete %s. Reason: %s' % (file_path, e))

# Dataset directories
train_dir = os.path.join(dataset_dir, "train2017")
annotations_dir =  os.path.join(dataset_dir, "annotations")
val_dir = os.path.join(dataset_dir, "val2017")

# Clear dataset directory if CLEAN_DATASET_DIR is Yes
if clean_dataset_dir == 'Yes':
   logging.warning('Cleaning up dataset directory')
   clear_directory(train_dir)
   clear_directory(annotations_dir)
   clear_directory(val_dir)

if recreate_annotations == 'Yes':
   logging.warning('Cleaning up annotations')
   clear_directory(annotations_dir)

#Create dataset directory structure
logging.info('Create directory structure if not exist here: '+str(dataset_dir))

for directory in annotations_dir,train_dir,val_dir:
  if not os.path.exists(directory):
    os.makedirs(directory)

# Check if the required files exist in the annotations_dir
required_files = ["instances_train2017.json", "instances_val2017.json"]
files_exist = all(os.path.exists(os.path.join(annotations_dir, file)) for file in required_files)

# Download the annotations file only if the required files do not exist
if not files_exist:
  annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
  annotations_file = os.path.join(annotations_dir, "annotations_trainval2017.zip")

  logging.info('Start download annotations')
  wget.download(annotations_url, annotations_file)

  logging.info('Start zipping out annotations')
  with zipfile.ZipFile(annotations_file, 'r') as zip_ref:
    zip_ref.extractall(dataset_dir)

  os.remove(annotations_file)

# Delete files in the zip that do not start with "instances_"
  file_list = os.listdir(annotations_dir)
  for file_name in file_list:
    if not file_name.startswith("instances_"):
      file_path = os.path.join(annotations_dir, file_name)
      os.remove(file_path)
else:
  logging.info('Skipping the download of annotations, they already exist')

# Download validation pictures
if len(os.listdir(val_dir)) != 5000:

  # Clear out the target directory
  clear_directory(val_dir)

  # Start downloading
  logging.info('Start download validation pictures')
  images_url = "http://images.cocodataset.org/zips/val2017.zip"
  images_file =  os.path.join(val_dir,"val2017.zip")
  wget.download(images_url, images_file)

  logging.info('Start zipping out validation')
  with zipfile.ZipFile(images_file, 'r') as zip_ref:
    zip_ref.extractall(dataset_dir)

  os.remove(images_file)

  logging.info('Validation pictures downloaded')

else:
  logging.info('Skipping the download of validation pictures, they already exist')

#Download full set of train pictures (We always download because is much faster to download all together in a zip than separately
#  and we can set the size of the dataset more flexibly)

if len(os.listdir(train_dir)) !=118287:

    # Clear out the target directory
    clear_directory(train_dir)

    # Start downloading
    logging.info('Start download train pictures')
    images_url = "http://images.cocodataset.org/zips/train2017.zip"
    images_file = os.path.join(train_dir,"train2017.zip")
    wget.download(images_url,images_file)

    logging.info('Start zipping out train pictures')
    with zipfile.ZipFile(images_file, 'r') as zip_ref:
      zip_ref.extractall(dataset_dir)

    os.remove(images_file)

else:
    logging.info('Skipping the download of train pictures, they already exist')

# Create a new annotation with subset of train images if download_percent is not 100
if download_percent != 100:

  #Initalize COCO Dataset
   logging.info('Use subset of train images mode')
   logging.info('Initialize COCO dataset')
   train_annotation_path = f'{annotations_dir}/instances_train2017.json'
   coco_train = COCO(train_annotation_path)

   # Get all category names and IDs
   categories = coco_train.loadCats(coco_train.getCatIds())

   #category_names = [category['name'] for category in categories]
   category_id_to_name = {category['id']: category['name'] for category in categories}
   category_ids = [category['id'] for category in categories]

   # Get image IDs
   image_ids_train = coco_train.getImgIds()
   num_image_train = len(image_ids_train)

   # Dict for new annotation file
   new_anns = {
    'info': coco_train.dataset['info'],
    'licenses': coco_train.dataset['licenses'],
    'images': [],
    'annotations': [],
    'categories': coco_train.dataset['categories']
   }

   # Download subset of images per category
   for category_id in category_ids:

     # Get image IDs for the current category
     image_ids = coco_train.getImgIds(catIds=category_id)

     # Calculate the number of images to download
     image_download_num = int(len(image_ids) * (download_percent) / 100)
     #image_download_num = 3

     # Select a subset of image IDs
     download_image_ids = image_ids[:image_download_num]

     logging.info(f"Category: {category_id_to_name.get(category_id)}, Full dataset size: {len(image_ids)} image, New size: {len(download_image_ids)} image")

     # Download the images
     for image_id in download_image_ids:
         image_info = coco_train.loadImgs(image_id)[0]
         image_file_name = image_info['file_name']
         image_url = f"http://images.cocodataset.org/train2017/{image_file_name}"
         image_file =  os.path.join(train_dir,image_file_name)

         # Add to new annotations file
         new_anns['images'].append(image_info)
         ann_ids = coco_train.getAnnIds(imgIds=image_id)
         anns = coco_train.loadAnns(ann_ids)
         for ann in anns:
           new_anns['annotations'].append(ann)

# Write out the new annotation file
with open(train_annotation_path, 'w') as f:
    json.dump(new_anns, f)
    logging.info('New annotation file created')

logging.info('Downloading process finished')
