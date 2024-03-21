from PIL import Image # Importing the Image module from the Python Imaging Library (PIL)
import os # Importing the os module for file and directory operations

# Check if this script is being run directly
if __name__ == '__main__':
    print('Run main.py first!')

class cleaning_images:

    def resize_image(final_size, im):

        """
        Resize the given image to fit within a square of the specified size.
        final_size: The size of the square to fit the image into
        im: The image to be resized
        """

        size = im.size # Get the size of the original image
        ratio = float(final_size) / max(size) # Calculate the resizing ratio
        new_image_size = tuple([int(x*ratio) for x in size]) # Calculate the new image size
        im = im.resize(new_image_size, Image.LANCZOS) # Resize the image using Lanczos resampling
        new_im = Image.new("RGB", (final_size, final_size)) # Create a new blank image of the final size
        new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2)) # Paste the resized image onto the blank image, centered
        return new_im

    def process_images(self):

        """
        Process images in a directory by resizing them and saving the resized versions.
        """
        
        path = "images_fb/images/" # Path to the directory containing the original images
        dirs = os.listdir(path) # Get a list of files in the directory
        final_size = 256 # Final size of the images after resizing
        
        # Iterate through the files in the directory
        for n, item in enumerate(dirs, 1):
            filename, extension = os.path.splitext(item) # Split the filename and extension
            im = Image.open('images_fb/images/' + item) # Open the image file
            new_im = cleaning_images.resize_image(final_size, im) # Resize the image
            new_im.save(f'cleaned_images/{filename}{extension}') # Save the resized image with the same filename in a different directory