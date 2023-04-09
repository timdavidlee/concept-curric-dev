# Download Olist dataset via kaggle API

## 1. Install the kaggle command line
```sh
pip install kaggle
```

## 2. get a kaggle API key - go to Kaggle, create an account, under settings, create an API key save somewhere.

Then once the directory is made, create the creds file in a hidden directory 

```sh
mkdir ~/.kaggle
vi ~/.kaggle/kaggle.json
# verify the changes have been saved
cat ~/.kaggle/kaggle.json
# change the file to be readonly
chmod 600 ~/.kaggle/kaggle.json
```

## 3. then download the dataset

Verify the kaggle cli is there

```sh
kaggle --version
# Kaggle API 1.5.13
```

Then use the `datasets` subcommand

```
kaggle datasets --help

optional arguments:
  -h, --help            show this help message and exit

commands:
  {list,files,download,create,version,init,metadata,status}
    list                List available datasets
    files               List dataset files
    download            Download dataset files
    create              Create a new dataset
    version             Create a new dataset version
    init                Initialize metadata file for dataset creation
    metadata            Download metadata about a dataset\
```

```sh
kaggle datasets download olistbr/brazilian-ecommerce
# Downloading brazilian-ecommerce.zip to /home/tlee
```

Once the above is run, it should tell where its being saved

```
sudo apt install -y unzip
unzip brazilian-ecommerce.zip
```

```
Archive:  brazilian-ecommerce.zip
  inflating: olist_customers_dataset.csv  
  inflating: olist_geolocation_dataset.csv  
  inflating: olist_order_items_dataset.csv  
  inflating: olist_order_payments_dataset.csv  
  inflating: olist_order_reviews_dataset.csv  
  inflating: olist_orders_dataset.csv  
  inflating: olist_products_dataset.csv  
  inflating: olist_sellers_dataset.csv  
  inflating: product_category_name_translation.csv  
```