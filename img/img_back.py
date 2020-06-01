from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

img = Image.open("/home/luyizhou/hsn_v1/out/01_tuning_patch/glas/vertical/output.png.png")
resize = transforms.Resize([828, 636])
img = resize(img)
img.save('/home/luyizhou/hsn_v1/img/output_test.png')