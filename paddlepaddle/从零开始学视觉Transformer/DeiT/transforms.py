import numpy as np
import paddle
import paddle.vision.transforms as T
from PIL import Image

class Resize():
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return T.resize(image, self.size)

class CenterCrop():
    def __init__(self, size):
        self.size = size

    def crop(self, image, region):
        cropped_image = T.crop(image, *region)
        return cropped_image

    def __call__(self, image):
        h, w = image.size
        ch, cw = self.size
        crop_top = int(round(h-ch) / 2)
        crop_left = int(round(w-cw) / 2)
        return self.crop(image, (crop_top, crop_left, ch, cw))

class ToTensor():
    def __init__(self):
        pass

    def __call__(self, image):
        img = paddle.to_tensor(np.array(image)) # hwc
        if img.dtype == paddle.uint8:
            img = paddle.cast(img, 'float32') / 255.0
        img = img.transpose([2, 0, 1]) # chw
        return img

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

def main():
    img = Image.open('image.jpg')
    transforms = Compose([Resize([256, 256]),
                          CenterCrop([112, 112]),
                          ToTensor()])
    out = transforms(img)
    print(out)
    print(out.shape)

if __name__ == '__main__':
    main()