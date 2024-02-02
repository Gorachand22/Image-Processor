import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import plotly.graph_objects as go
import plotly.express as px

class Work:
    def __init__(self):
        pass
    def show_images(self, original, processed):
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption='Original Image', width= 300)
        with col2:
            st.image(processed, caption='Processed Image.', width= 300)

    def load_image(self):
        """
        Function to load an image file using the streamlit file uploader. 
        Returns a numpy array of the image if the file is uploaded, otherwise returns None.
        """
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"], accept_multiple_files= False)
        if uploaded_file is not None:
            return np.array(Image.open(uploaded_file))

    def load_image0(self):
        """
        Function to load an image file from the user and return it as a NumPy array.
        """
        new_file = st.file_uploader("Choose another image file", type=["png", "jpg", "jpeg"], accept_multiple_files= False)
        if new_file is not None:
            return np.array(Image.open(new_file))



    def details_of_an_image(self,img):
        """
        Function to display details of an image including its shape, height, width, type, and size.

        Parameters:
        img : numpy.ndarray
            The image to display details for.

        Returns:
        None
        """
        st.write("Details of an image\n")
        st.image(img, caption='Uploaded Image.', width= 300)
        st.write(f"Shape of the image: ", img.ndim)
        st.write(f"Shape of the image: ", img.shape)
        st.write(f"Height of the image: ", img.shape[0])
        st.write(f"Width of the image: ", img.shape[1])
        st.write(f"Type of the image: ", img.dtype)
        st.write(f"Size of the image: ", img.size)
        
        self.save_image(img)

    def rotating_an_image(self,img, angel):
        """
        Rotate an image by the given angle.
        
        Parameters:
        img (image): The input image to be rotated.
        angel (int): The angle of rotation in degrees.
        
        Returns:
        None
        """
        st.write("Rotating an image")
        new_img = img.copy()
        n = None
        if angel == 90:
            n = 1
        elif angel == 180:
            n = 2
        elif angel == 270:
            n = 3
        else:
            n = 0

        col1,col2 = st.columns(2)
        with col1:
            st.image(img, caption='Uploaded Image.', width= 300)
        with col2:
            for i in range(n):
                new_img = np.rot90(new_img)
            st.image(new_img, caption='Rotated Image.', width= 300)
        
        self.save_image(new_img)
    
    def negative_of_an_image(self,img):
        """
        Function to compute the negative of an image and display the original and negative images side by side.
        
        Parameters:
        img (numpy.ndarray): The input image as a NumPy array.

        Returns:
        None
        """
        st.write("Negative of an image")
        new_img = 255 - img
        col1, col2 = st.columns(2)
        with col1:
            st.write("Uploaded Image")
            st.image(img, caption='Uploaded Image.', width= 300)
        with col2:
            st.write("Negative Image")
            st.image(new_img, caption='Negative Image.', width= 300)
        
        self.save_image(new_img)
    
    def trim_image(self,img):
        """
        Trim the input image based on the specified upper and lower length and width values from the sidebar sliders.
        Display the trimmed image and save it.
        """
        st.write("Trim Image")
        st.sidebar.slider
        UL = st.sidebar.slider("Upper Length", 0, img.shape[0]//2)
        LL= st.sidebar.slider("Lower Length", 0, img.shape[0]//2)
        UW= st.sidebar.slider("Upper Width", 0, img.shape[1]//2)
        LW = st.sidebar.slider("Lower Width", 0, img.shape[1]//2)

        trimmed_image = img[UL:img.shape[0]+1 - LL, UW:img.shape[1]+1 - LW, 0:img.shape[2]+1]
        
        st.image(trimmed_image, caption='Trimmed Image', width= 300)
        
        self.save_image(trimmed_image)

    def flip_image(self, img, flip):
        """
        Flip the input image vertically or horizontally based on the 'flip' parameter.

        Parameters:
        img (numpy.ndarray): The input image to be flipped.
        flip (str): The direction in which to flip the image. Can be "[up/down]" or "[left/right]".

        Returns:
        None
        """
        st.write("Flip Image")

        if flip == "[up/down]":
            new_img = np.flipud(img)
        elif flip == "[left/right]":
            new_img = np.fliplr(img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption='Uploaded Image.', width= 300)
        with col2:
            st.image(new_img, caption='Flipped Image.', width= 300)
        
        self.save_image(new_img)

    def normal_mode_blend(self,img, img0):
        """
        Blend two images in normal mode and ensure they have the same dimensions for blending.
        
        Args:
        - img: The first input image
        - img0: The second input image
        
        Returns:
        - img: The first input image after padding (if needed) for blending
        - img0: The second input image after padding (if needed) for blending
        """
        st.write("Normal Mode")
        try:
            if img.shape[2] != img0.shape[2]:
            # If img0 has 4 channels (RGBA), convert it to 3 channels (RGB) by discarding the alpha channel
                img_3 = img.shape[2]
                img0_3 = img0.shape[2]
                if img_3 > img0_3:
                    img = img[:, :, :img0_3]
                else:
                    img0 = img0[:, :, :img_3]
        except:
            pass

        # Ensure img0 has the same dimensions as img for blending
        if img.shape[0] != img0.shape[0] or img.shape[1] != img0.shape[1]:
            max_height = max(img.shape[0], img0.shape[0])
            max_width = max(img.shape[1], img0.shape[1])

            pad_height_img = max_height - img.shape[0]
            pad_width_img = max_width - img.shape[1]
            pad_height_img0 = max_height - img0.shape[0]
            pad_width_img0 = max_width - img0.shape[1]

            img = np.pad(img, ((0, pad_height_img), (0, pad_width_img), (0, 0)), mode='constant')
            img0 = np.pad(img0, ((0, pad_height_img0), (0, pad_width_img0), (0, 0)), mode='constant')

            # Blend the images
        return img, img0

    def resizing_mode_blend(self,img, img0):
        """
        Resizes two images to have the same dimensions based on the dimensions of the smaller image in each dimension.
        
        Parameters:
        img: ndarray
            The first input image to be resized.
        img0: ndarray
            The second input image to be resized.
        
        Returns:
        img: ndarray
            The resized first input image.
        img0: ndarray
            The resized second input image.
        """
        st.write("Resizing Mode")
        try:
            if img.shape[2] != img0.shape[2]:
            
                img_3 = img.shape[2]
                img0_3 = img0.shape[2]
                if img_3 > img0_3:
                    img = img[:, :, :img0_3]
                else:
                    img0 = img0[:, :, :img_3]
        except:
            pass
            
        if img.shape[0] != img0.shape[0]:
            img_0 = img.shape[0]
            img0_0 = img0.shape[0]
            if img_0 > img0_0:
                img = img[:img0_0, :, :]
            else:
                img0 = img0[:img_0, :, :]
            
        if img.shape[1] != img0.shape[1]:
            img_1 = img.shape[1]
            img0_1 = img0.shape[1]
            if img_1 > img0_1:
                img = img[:, :img0_1, :]
            else:
                img0 = img0[:, :img_1, :]
        
        return img, img0

    def show_blend(self, img, img0):
        """
        Show a blended image by combining img and img0 with a 60/40 ratio and displaying 
        the original images and the blended image in a 3-column layout. Finally, save the 
        blended image.
        """
        new_img = (img * 0.6 + img0 * 0.4).astype(np.uint8)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, caption='Image 1.', width= 300)
        with col2:
            st.image(img0, caption='Image 2.', width= 300)
        with col3:
            st.image(new_img, caption='Blended Image.', width= 300)
            

        self.save_image(new_img)

    def blending_two_images(self, img, img0):
        """
        Blend two images together and display the result in the Streamlit app sidebar.
        
        Parameters:
        img (Image): The first image to be blended.
        img0 (Image): The second image to be blended.

        Returns:
        None
        """
        st.write("Blending two images")

        st.sidebar.write("""
                - Some Point To Remember
                - If Your Two Images are not is same shape then it will convert it to same shape automatically
                - But if don't want to do auto resize and then do it manually
                - For that you have to press Normal mode button in sidebar
                - Both images should be in same format like (3D AND 3D) or (2D AND 2D)
                - Not like one is in 3D and another is in 2D
                """)

        if st.sidebar.button("Normal Mode"):
                img2, img3 = self.normal_mode_blend(img, img0)
                img = img2
                img0 = img3
                self.show_blend(img, img0) # Blend the images  
        
        if st.sidebar.button("Resizing Mode"):
                img2, img3 = self.resizing_mode_blend(img, img0)
                img = img2
                img0 = img3
                self.show_blend(img, img0)

    def add_frame_to_an_image(self,img):
        """
        Add a frame to the input image using the specified upper and lower length and width. 
        Allows the user to choose the color of the frame and displays the original and framed images.
        """
        st.write("Add Frame to an image")
        if img.shape[2]>3:
            img = img[:,:,:3]
        st.sidebar.slider
        UL = st.sidebar.slider("Upper Length", 0, 30)
        LL = st.sidebar.slider("Lower Length", 0, 30)
        UW = st.sidebar.slider("Upper Width", 0, 30)
        LW = st.sidebar.slider("Lower Width", 0, 30)

        colors_rgb = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'black': (0, 0, 0),
            'pink': (255, 192, 203),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'brown': (165, 42, 42),
            'white': (255, 255, 255),
            'gray': (128, 128, 128)
        }

        choose = st.sidebar.radio("Choose Color", list(colors_rgb.keys()))
        if choose is not None:
            color = colors_rgb[choose]
            bordered_img = cv2.copyMakeBorder(img, UL, LL, UW, LW, cv2.BORDER_CONSTANT, value=color)
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption='Original Image', use_column_width=True)
            with col2:
                st.image(bordered_img, caption='Image with Frame', use_column_width=True)

            self.save_image(bordered_img)

    def rgb_channels(self, img):
        """
        Function to separate the RGB channels of the input image and display them using Streamlit.
        Parameters:
        - img: Input image in the form of a numpy array.

        Returns:
        This function does not return any value.
        """
        st.write("RGB Channels")
        img_R, img_G, img_B = img.copy(), img.copy(), img.copy()
        img_R[:, :, (1, 2)] = 0
        img_G[:, :, (0, 2)] = 0
        img_B[:, :, (0, 1)] = 0
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_R, caption='Red Image', use_column_width=True)
        
        with col2:
            st.image(img_G, caption='Green Image', use_column_width=True)
            
        with col3:
            st.image(img_B, caption='Blue Image', use_column_width=True)
        d = {
            "Red": img_R,
            "Green": img_G,
            "Blue": img_B
        }
        ask = st.sidebar.radio("Download Images", ("Red", "Green", "Blue"))
        if ask is not None:
            self.save_image(d[ask])


    def pasting_with_slice(self, img, img0):
        """
        Function to paste a slice of an image onto another image.

        Args:
        - img: The source image to be pasted onto another image
        - img0: The destination image onto which the source image will be pasted

        Returns:
        - None
        """
        st.write("We can paste a slice of an image onto another image.")
        if img.shape[0] != img0.shape[0] or img.shape[1] != img0.shape[1]:
            img2,img3 = self.resizing_mode_blend(img, img0)
            img = img2
            img0 = img3

        dst_copy = img0.copy()
        roller = st.sidebar.slider("roller", 1, np.max(img.shape[:2])//6)
        try:
            dst_copy[roller*2:roller*4, roller*4:roller*6] = img[roller:roller*3, roller:roller*3]
            st.image(dst_copy, caption='Sliced Image', width= 300)
        except:
            pass

    def  binarize_image(self, img0):
        """
        Binarize the input image based on the selected binary form and display the original 
        and binary images using Streamlit. Also, attempts to save the binary image.
        """
        st.write("Binarize Image")
        choice = st.sidebar.radio("Select Binary Form", ("16", "32", "64", "128", "192"))
        img_16 = (img0 > 16) * 255
        img_32 = (img0 > 32) * 255
        img_64 = (img0 > 64) * 255
        img_128 = (img0 > 128) * 255
        img_192 = (img0 > 192) * 255
        d = {
            "16":img_16,
            "32":img_32,
            "64":img_64,
            "128":img_128,
            "192":img_192
        }
        if choice is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img0, caption='Original Image', width= 300)
            with col2:
                st.image(d[choice], caption='Binary Image', width= 300)
        try:
            self.save_image(d[choice])
        except:
            pass

    def pixel_intensity(self,img):
        """
        Calculate and display the histogram of pixel intensities in the given image.

        Parameters:
        img (numpy.ndarray): The input image as a NumPy array.

        Returns:
        None
        """
        st.write("Pixel Intensity")
        fig = px.histogram(
            x=img.ravel(),
            nbins=200,
            range_x=[0, 256],
            title="Histogram",
            opacity=0.8,
            color_discrete_sequence=["red"],
            labels=dict(x="Pixel Intensity", y="Count"),
            template="plotly_white",
            orientation="v"
        )
        st.header("Histogram Plot")
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    def color_reduction(self, img0):
        """
        Function for color reduction of an image.

        Args:
        img0: The input image to be processed.

        Returns:
        None
        """
        st.write("Color Reduction")
        roller = st.sidebar.slider("Select Binary Form", 8, 200, 8, 8)
        new_img = (img0 // roller) * roller
    
        if roller is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img0, caption='Original Image', width= 300)
            with col2:
                st.image(new_img, caption='Binary Image', width= 300)
        try:
            self.save_image(new_img)
        except:
            pass
    
    def color_enhancement(self,img):
        """
        Perform color enhancement on the input image.

        Parameters:
        img: numpy.ndarray
            The input image to be color enhanced.

        Returns:
        None
        """
        st.write("Color Enhancement")
        saturation_factor = st.sidebar.slider("Saturation Factor", 0.1, 3.0, 1.0, 0.1)

        # Convert the image to HSV color space
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Multiply the saturation channel by the saturation factor
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation_factor, 0, 255)

        # Convert the image back to RGB color space
        enhanced_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption='Original Image', width=300)
        with col2:
            st.image(enhanced_img, caption='Enhanced Image', width=300)

        try:
            self.save_image(enhanced_img)
        except:
            pass

    def save_image(self, img):
        """
        Save the given image to a PNG file and create a download button for it.

        Parameters:
        img: numpy array
            The image to be saved.

        Returns:
        None
        """
        img = Image.fromarray(img)
        # Convert PIL Image to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        # Create a download button
        st.download_button(label="Save Image", data=img_bytes, file_name='processed_image.png', mime='image/png', key=None)



