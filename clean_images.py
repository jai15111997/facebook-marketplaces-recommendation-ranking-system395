from PIL import Image
import os

if __name__ == '__main__':
    print('Run main.py first!')

class cleaning_images:
    def resize_image(final_size, im):
        size = im.size
        ratio = float(final_size) / max(size)
        new_image_size = tuple([int(x*ratio) for x in size])
        im = im.resize(new_image_size, Image.LANCZOS)
        new_im = Image.new("RGB", (final_size, final_size))
        new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
        return new_im

    def process_images(self):
        path = "images_fb/images/"
        dirs = os.listdir(path)
        final_size = 256
        for n, item in enumerate(dirs, 1):
            filename, extension = os.path.splitext(item)
            im = Image.open('images_fb/images/' + item)
            new_im = cleaning_images.resize_image(final_size, im)
            new_im.save(f'cleaned_images/{filename}{extension}')