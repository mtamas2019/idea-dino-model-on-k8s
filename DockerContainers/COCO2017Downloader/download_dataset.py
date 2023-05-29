import wget, json, zipfile, os, logging, shutil

#Variables
dataset_dir = os.environ.get('DATASET_DIR', 'COCODIR')

#Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)

#Functions
def clear_directory(target_dir):
    """Clears out a directory by removing all files inside. """
    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            logging.error('Failed to delete %s. Reason: %s' % (file_path, e))

#Create dataset directory structure
annotations_dir =  os.path.join(dataset_dir, "annotations")
train_dir = os.path.join(dataset_dir, "train2017")
val_dir = os.path.join(dataset_dir, "val2017")

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

# Download train pictures
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

logging.info('Script finished')
