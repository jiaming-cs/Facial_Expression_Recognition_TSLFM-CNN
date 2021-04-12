# Facial_expression_Recognition_TSLFM-CNN

## Overview

## Setput Environment

* Download Models  
  * Download models.zip from [google drive](https://drive.google.com/file/d/1M88E87IjWGMVJcRn69TEO8KG4GlMMfiH/view?usp=sharing), unzip it to root folder of the repository

* Create Virtual Environment

```bash
conda create -n venv python=3.7
```

* Activate virtual environment

```bash
conda activate venv 
```

* Install dependencies  

```bash
pip install -r requirements.txt
```

## Install Codec Pack

* On Windows, you need to install [K-Lite Codec Pack](https://codecguide.com/download_kl.htm) to play the video.

## Run Server  

```bash
cd web
python server.py
```

## Run Application

Open another terminal  

```bash
cd {repository_root_folder}
conda activate venv
python app.py
```

