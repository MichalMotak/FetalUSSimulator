import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import simplejpeg

from PIL import Image
import io


# Read Images
img = mpimg.imread('imageRGBpng.png')
 
# Output Images
plt.imshow(img)
print(img.shape)


# plt.show()

# print(simplejpeg.is_jpeg(simplejpeg.encode_jpeg(img)))


# =====================  WORKS

in_file = open("bytesRGB_png.txt", "rb") # opening for [r]eading as [b]inary
data = in_file.read() # if you only wanted to read 512 bytes, do .read(512)
in_file.close()

print(data[:50])
print(len(data))
# print(len(str(data, 'UTF-8')))

image = Image.open(io.BytesIO(data))
image.show()

# ======================= WORKS 


# image = Image.frombytes('RGBA', (232,211), data, 'raw')


# in_file = open("bytesRGB.txt", "rb") # opening for [r]eading as [b]inary
# data = in_file.read() # if you only wanted to read 512 bytes, do .read(512)
# in_file.close()

# print(data[:50])
# print(len(data))
