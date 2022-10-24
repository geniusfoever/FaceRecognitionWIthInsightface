# Alignment
### Using Insightface Default Detection
src/align/align_my.py
--indir D:\DataBase\51\lfw --outdir D:\DataBase\51\lfw_output --image-size 112,112

# Make .list File
src/data/face2rec_my.py
--list --root D:/DataBase/51/lfw_output/ --prefix D:/DataBase/51/lfw_output/

# Make .rec File
Same python file as above
--root D:/DataBase/51/lfw_output/ --prefix D:/DataBase/51/lfw_output**[.lst]**

# .lst .rec to Image Folders
script/rec2img_my.py

# Add Mask
MaskTheFace/mask_the_face.py
--path D:/database/51/lfw_output --outpath D:/database/51/lfw_masked --mask_type surgical --verbose --write_original_image