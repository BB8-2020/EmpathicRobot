# Data

In this folder you can find all processing that has been done to our datasets. 
Our processed sets can be found in our [Onedrive](https://hogeschoolutrecht-my.sharepoint.com/personal/maria_dukmak_student_hu_nl/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly9ob2dlc2Nob29sdXRyZWNodC1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9tYXJpYV9kdWttYWtfc3R1ZGVudF9odV9ubC9FdHBQYlBYWjdrVkh1WW1RTWlwNTJwc0JkS1kxUWNWa0d1aS1teXBMQ0xZdjh3P3J0aW1lPWdac0I1SzM5MkVn&id=%2Fpersonal%2Fmaria%5Fdukmak%5Fstudent%5Fhu%5Fnl%2FDocuments%2FBB8%2Fprocessed%5Fsets).
For acces to this onedrive you can request access by mailing to [maria.dukmak@student.hu.nl](maria.dukmak@student.hu.nl)

## Installation
If you want to prepare the data yourself you need to download the files from their source.

### FerPlus
You can download the `Fer2013` dataset 
[here](https://www.kaggle.com/deadskull7/fer2013). 
In order to use the `FerPlus` dataset you must download the 
[FerPlus labels](https://github.com/microsoft/FERPlus) and 
overwrite the `Fer2013` labels with these.

If you have downloaded all data you can run [this](https://github.com/BB8-2020/EmpathicRobot/blob/data/data/ferPlus/data_processing_FerPlus.ipynb)
notebook to merge the two dataframes into one frame and analyse, process and prepare the data for our model.
For all functions that the notebook is using you need the [ferPlus_functions.py](https://github.com/BB8-2020/EmpathicRobot/blob/data/data/ferPlus/ferPlus_functions.py)
file containing `ferPlus` functions only, and the [general_definitions.py](https://github.com/BB8-2020/EmpathicRobot/blob/data/data/general_defenitions.py) 
file consisting of all functions that are needed in the `FerPlus` and the `AffectNet` processing.

### AffectNet
In order to download the `AffectNet` dataset you have to get permission in front. 
You can request access [here](http://mohammadmahoor.com/affectnet/).

If you have downloaded the tar files from the source you can run [this](https://github.com/BB8-2020/EmpathicRobot/blob/data/data/ferPlus/data_processing_FerPlus.ipynb) 
notebook to merge the two dataframes into one frame and to analyse, process and prepare the data for our model.
For all functions that the notebook is using you need the [affectNet_functions.py](https://github.com/BB8-2020/EmpathicRobot/blob/data/data/affectNet/affectNet_functions.py)
file consisting of all `AffectNet` functions, and the [general_definitions.py](https://github.com/BB8-2020/EmpathicRobot/blob/data/data/general_defenitions.py) 
file consisting of all functions that are needed in the `FerPlus` and the `AffectNet` processing.
