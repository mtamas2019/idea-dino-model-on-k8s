import wget, json, zipfile, os, logging

#Variables
dataset_dir = os.environ.get('DATASET_DIR', 'COCODIR')

#Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Create dataset directory structure
annotations_dir =  os.path.join(dataset_dir, "annotations")
train_dir = os.path.join(dataset_dir, "train2017")
val_dir = os.path.join(dataset_dir, "val2017")

for directory in annotations_dir,train_dir,val_dir:
  if not os.path.exists(directory):
    os.makedirs(directory)

logging.info('Directory structure created here: {}'.format(str(dataset_dir)))

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


# Download validation pictures
if len(os.listdir(val_dir)) != 5000:
  logging.info('Start download validation pictures')
  images_url = "http://images.cocodataset.org/zips/val2017.zip"
  images_file =  os.path.join(val_dir,"val2017.zip")
  wget.download(images_url, images_file)

  logging.info('Start zipping out validation')
  with zipfile.ZipFile(images_file, 'r') as zip_ref:
    zip_ref.extractall(dataset_dir)

  os.remove(images_file)

# Download train pictures
if len(os.listdir(train_dir)) !=118287:
  logging.info('Start download train pictures')
  images_url = "http://images.cocodataset.org/zips/train2017.zip"
  images_file = os.path.join(train_dir,"train2017.zip")
  wget.download(images_url,images_file)

  logging.info('Start zipping out train pictures')
  with zipfile.ZipFile(images_file, 'r') as zip_ref:
    zip_ref.extractall(dataset_dir)

  os.remove(images_file)

logging.info('Script finished')
