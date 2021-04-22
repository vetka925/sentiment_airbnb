mkdir %USERPROFILE%\.kaggle
mkdir data
copy .\kaggle.json C:\Users\vitaliy\.kaggle\
kaggle datasets download -d labdmitriy/airbnb -p .\data\airbnb
kaggle datasets download -d datafiniti/hotel-reviews -p .\data\hotel-reviews
kaggle datasets download -d jiashenliu/515k-hotel-reviews-data-in-europe -p .\data\hotel-reviews-eu
python -m spacy download en_core_web_sm