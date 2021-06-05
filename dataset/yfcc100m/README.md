# Download YFCC100M
This is the directory to download image and metadata of YFCC100M.
## Folder Structure
```
dataset/yfcc100m/
├── yfcc100m_subset_data.tsv - lines, photoid, hash
├── new_metadata.csv - photoid, hash, userid, datetaken, dateuploaded, capturedevice, title, description, usertags, machinetags, longitude, latitude, accuracy, licensename, marker, year, yearmonth, month, a_autotags, p_town, p_state, p_country, p_places
├── data/ - image files (run 0_download.py)
│   ├── 000.jpg
│   └── photoid.jpg
├── file_error.log - jpg download failure records (run 0_download.py)
├── dataset.tsv - photoid, description, url (run 0_download.py)
├── train.tsv - photoid, description, url (run 1_split.py)
├── dev.tsv - photoid, description, url (run 1_split.py)
├── 0_download.py - python file for download
└── 1_split.py - python file for split train/dev

```
## Usage
> Download subset-info/metadata
* Download subset info from OpenAI - [download link](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md)
* Download metadata from multimediacommons - [download link](https://drive.google.com/file/d/0B6D7jCorgVvCaTRNQmRBQ0UzcUk/view)
* Move all files into ```dataset/yfcc100m```
> Prepare dataset
```
python 0_download.py
```
> Split dataset
```
python 1_split.py
```
