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

# Image to pairs.txt
script/generate_pairs_txt.py

# pairs.txt to test.bin
script/mydata2pack.py

# Add Mask
MaskTheFace/mask_the_face.py
--path D:/database/51/lfw_output --outpath D:/database/51/lfw_masked --mask_type surgical --verbose --write_original_image

# Remove Last Layer fc7
deploy/model_slim.py
--model "/content/gdrive/My Drive/insightface/model/r100-arcface-emore/model",3

# mxnet to onnx
/InsightFace-REST/src/api_trt/modules/converters/
import insight2onnx
sym = '/content/gdrive/My Drive/insightface/model/r100-arcface-emore/models-symbol.json'
params = '/content/gdrive/My Drive/insightface/model/r100-arcface-emore/models-0000.params'
input_shape = [(1,3,112,112)]
    
onnx_file = '/content/gdrive/My Drive/insightface/model/output/r100-arcface-emore-lfw-mixed.onnx'
insight2onnx.convert_insight_model(sym, params, onnx_file)
  