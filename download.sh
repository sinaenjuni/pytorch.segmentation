
fileid="10dzQT-eiQF4oUMklBHINp6JeW_rlvoZF"
filename="CelebAMaskDataset.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

mkdir "data"
mv $filename "data/"
cd "data"
unzip $filename
rm $filename