# Empathic Robot
[![Build](https://github.com/BB8-2020/EmpathicRobot/actions/workflows/python-build.yml/badge.svg)](https://github.com/BB82020/EmpathicRobot/actions/workflows/python-build.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

This project is to recognize emotions using camera images of the Sanbot Elf robot.
 

## Installation 
Rename `.env.example` to `.env`  

To install al requirements by run
    
        pip install .

**Developer** 
If you are a developer set `developer=True` in `.env`


## Run 
By default the model is placed in the [Sanbot](https://github.com/BB8-2020/MoodGuesserSanbot) application.
To creat, run and test the model you colud run [this](https://github.com/BB8-2020/EmpathicRobot/blob/main/src/models/classification_model/conv/ferPlus_model.ipynb) file. 
Therefore you need to get the data by generate it yourself or download it from the [Onedrive](https://hogeschoolutrecht-my.sharepoint.com/personal/maria_dukmak_student_hu_nl/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmaria%5Fdukmak%5Fstudent%5Fhu%5Fnl%2FDocuments%2FBB8). Read [this](https://github.com/BB8-2020/EmpathicRobot/blob/main/src/data/README.md) for more info. 


## License
MIT
