"""This Is The Main File For The Image Processing Project"""
import streamlit as st
from work import Work
w = Work()

if "img" not in st.session_state:
    st.session_state["img"]= None

if "img0" not in st.session_state:
    st.session_state["img0"]= None

st.set_page_config(page_title="The Image Processing Project", page_icon=":bulb:", layout="wide")

st.title("The Image Processing Project")
st.header("Welcome To The Image Processing Project")

options = st.selectbox("Select an option",["Image Processor", "Demo", "About"])
if options == "Image Processor":
    
    choice = st.selectbox(
        'What would you like to do?',
        ('Details of an image', 'Rotating an image', 'Negative of an image', 'Trim Image', 
        "Blending two images", "Flipping an image", "Add Frame to an image", "RGB Channels",
        "Pasting With Slice","Binarize Image","Pixel Intensity",
        "Color Reduction of an image", "Color Enhancement of an image"))

    if choice == 'Details of an image':
        st.session_state["img"] = w.load_image()
        if st.session_state["img"] is not None:
            w.details_of_an_image(st.session_state["img"])
    
    elif choice == 'Rotating an image':
        st.session_state["img"] = w.load_image()
        if st.session_state["img"] is not None:
            w.rotating_an_image(st.session_state["img"], int(st.radio("Rotate", ["90", "180", "270"])))
    
    elif choice == 'Negative of an image':
        st.session_state["img"] = w.load_image()
        if st.session_state["img"] is not None:
            w.negative_of_an_image(st.session_state["img"])
        
    elif choice == 'Trim Image':
        st.session_state["img"] = w.load_image()
        if st.session_state["img"] is not None:
            w.trim_image(st.session_state["img"])
    
    elif choice == "Blending two images":
        st.session_state["img"] = w.load_image()
        st.session_state["img0"] = w.load_image0()
        if st.session_state["img"] is not None and st.session_state["img0"] is not None:
            w.blending_two_images(st.session_state["img"], st.session_state["img0"])

    elif choice == "Flipping an image":
        st.session_state["img"] = w.load_image()
        if st.session_state["img"] is not None:
            w.flip_image(st.session_state["img"], st.radio("Flip", ["[up/down]", "[left/right]"]))
    
    elif choice == "Add Frame to an image":
        st.session_state["img"] = w.load_image()
        if st.session_state["img"] is not None:
            w.add_frame_to_an_image(st.session_state["img"])
    
    elif choice == "RGB Channels":
        st.session_state["img"] = w.load_image()
        if st.session_state["img"] is not None:
            w.rgb_channels(st.session_state["img"])

    elif choice == "Pasting With Slice":
        st.session_state["img"] = w.load_image()
        st.session_state["img0"] = w.load_image0()
        if st.session_state["img"] is not None and st.session_state["img0"] is not None:
            w.pasting_with_slice(st.session_state["img"], st.session_state["img0"])
    
    elif choice == "Binarize Image":
        st.session_state["img"] = w.load_image()
        if st.session_state["img"] is not None :
            w.binarize_image(st.session_state["img"])
    
    elif  choice == "Pixel Intensity":
        st.session_state["img"] = w.load_image()
        if st.session_state["img"] is not None:
            w.pixel_intensity(st.session_state["img"])
    
    elif choice == "Color Reduction of an image":
        st.session_state["img"] = w.load_image()
        if st.session_state["img"] is not None:
            w.color_reduction(st.session_state["img"])
    
    elif choice == "Color Enhancement of an image":
        st.session_state["img"] = w.load_image()
        if st.session_state["img"] is not None:
            w.color_enhancement(st.session_state["img"])
    


elif options == "Demo":
    st.header("Here You can see How it look like after being processed")
    if st.button("details_of_an_image"):
        w.show_images("images/details_of_an_image1.png", "images/details_of_an_image2.png")
    
    if st.button("rotating_an_image"):
        w.show_images("images/rotating_an_image1.png", "images/rotating_an_image2.png")
    
    if st.button("negative_of_an_image"):
        w.show_images("images/details_of_an_image1.png", "images/negative_of_an_image.png")
    
    if st.button("trim_image"):
        w.show_images("images/details_of_an_image1.png", "images/trim_image.png")
    
    if st.button("flip_image"):
        w.show_images("images/details_of_an_image1.png", "images/flip_image.png")
    
    if st.button("blending_two_images"):
        w.show_images("images/details_of_an_image1.png", "images/show_blend.png")
    
    if st.button("add_frame_to_an_image"):
        w.show_images("images/details_of_an_image1.png", "images/add_frame_to_an_image.png")
    
    if st.button("rgb_channels"):
        w.show_images("images/details_of_an_image1.png", "images/rgb_channels.png")
    
    if st.button("pasting_with_slice"):
        w.show_images("images/Emma.png", "images/pasting_with_slice.png")
    
    if st.button("binarize_image"):
        w.show_images("images/Emma.png", "images/binarize_image.png")
    
    if st.button("pixel_intensity"):
        w.show_images("images/details_of_an_image1.png", "images/pixel_intensity.png")
    
    if st.button("color_reduction"):
        w.show_images("images/details_of_an_image1.png", "images/color_reduction.png")
    
    if st.button("color_enhancement"):
        w.show_images("images/details_of_an_image1.png", "images/color_enhancement.png")

elif options == "About":
    st.header("About")
    def show_instructions():
        st.markdown("## Instructions")
        st.write("Welcome to the Image Processing App!")
        st.write("This app allows you to perform various image processing operations.")
        st.write("To get started, follow these steps:")
        st.write("1. Upload an image using the file uploader.")
        st.write("2. Select an operation from the dropdown menu.")
        st.write("3. Adjust any parameters using the sliders or radio buttons.")
        st.write("4. View the processed image and save it if desired.")

# Add this function call at the beginning of your app
    show_instructions()
