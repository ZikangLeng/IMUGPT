rm -rf checkpoints
mkdir checkpoints
cd checkpoints
echo "Downloading"
gdown "https://drive.google.com/uc?id=1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8"
echo "Extracting"
tar xfzv t2m.tar.gz
echo "Cleaning"
rm t2m.tar.gz

echo -e "Downloading done!"