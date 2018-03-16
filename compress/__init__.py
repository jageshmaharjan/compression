from keras.applications.vgg16 import VGG16

model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()