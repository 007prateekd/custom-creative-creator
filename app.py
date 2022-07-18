import cv2
import json
import tempfile
import streamlit as st
from pathlib import Path

from jojo_gan import pretrained_style as face_stylizer
from face_toolbox import blend_face as face_blender
from agis_net import stylize_text as text_stylizer
from text_processing import insert_text as text_inserter


with open("config.json") as json_file:
    data = json.load(json_file)


@st.experimental_memo
def stylize_face(image, option, device):
    projection_opt = "restyle" if data[option]["projection_opt"] == "restyle" else "e4e"
    n_iter = data[option]["n_iter"]
    with st.spinner("Stylizing your image"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            fp = Path(tmp_file.name)
            fp.write_bytes(image.getvalue())
            file_path = tmp_file.name
            style = data[option]["path"].split("/")[1].split(".")[0]
            my_output = face_stylizer.main(file_path, style, projection_opt, device, n_iter)
            my_output = my_output.cpu().permute(1, 2, 0).detach().numpy()
    # st.write("Your stylized face is displayed below")
    # st.image(my_output, width=400)
    my_output *= 255
    return my_output


def blend_face(option, src):  
    with st.spinner("Blending your face"):
        dst_path = data[option]["path_textless"]
        dst_h2c = data[option]["dst_h2c"]
        rot_degrees = data[option]["rot_degrees"] 
        center = data[option]["center"]
        res = face_blender.main(src, dst_path, dst_h2c, rot_degrees, center)
    # st.write("Your face has been inserted!")
    # st.image(res)
    return res


def stylize_text(font, text, direction, device):
    with st.spinner("Applying style to your text"):
        texture_name = data[font]["path"].split("/")[1].split(".")[0]
        few_size = 5
        n_encode = 4
        horizontal = True if direction == "Horizontal" else False
        padding = 2
        space_size = 10
        str_image = text_stylizer.main(texture_name, few_size, n_encode, horizontal, text, padding, space_size, device)
    # st.write("Your stylized text is displayed below")
    # st.image(str_image)
    return str_image


def insert_text(creative, str_image, cx, cy, scale_factor):
    with st.spinner("Inserting your text into the creative"):
        final_image = text_inserter.main(creative[..., ::-1], str_image, cx, cy, scale_factor)
        return final_image


def main(device):
    # usage
    st.sidebar.markdown("# Usage")
    st.sidebar.markdown("1. Select a creative from the given choices. Next, you can either stylize your face, \
    your choice of text or both.")
    st.sidebar.markdown("2. To stylize your face choose an image from your computer or take a picture using your webcam. ")
    st.sidebar.markdown("3. To stylize your choice of text, select a font, and then enter some text and choose text direction.  \n \
    **Note**: While the position of text and scaling factor is automatically filled for the best results, \
    you could also provide these values manually.")
    st.sidebar.markdown("4. Click on `Run` to generate your results.")
    # title
    st.title("C3: Custom Creative Creator")
    # select creative
    option = st.selectbox(
        "Select a creative",
        (None, "Creative 1", "Creative 2")
    )
    # recommended values
    recom_x = recom_y = 0
    recom_sf = 1.0
    if option:
        st.image(data[option]["path"])
        recom_x, recom_y = data[option]["recom_pos"]
        recom_sf = data[option]["recom_sf"]

    col1, col2 = st.columns([2, 1])
    flag1, flag2 = False, False
    # upload image
    uploaded_file = col1.file_uploader("Choose an image", disabled=flag1)
    if uploaded_file:  
        flag2 = True
        col1.image(uploaded_file, use_column_width="always")
    # or take image from camera
    expander = col1.expander("Or click here to capture")
    camera = expander.camera_input("", disabled=flag2)
    if camera:
        flag1 = True
    # select font
    font = col2.selectbox(
        "Select a font",
        (None, "Font 1", "Font 2")
    )
    text = None
    if font:
        col2.image(data[font]["path"], use_column_width="always")
        text = col2.text_input("Enter text", "", key=1)
        center = col2.text_input("Enter the position of text", f"{recom_x}, {recom_y}", key=2)
        cx, cy = tuple(map(int, center.split(",")))
        scale_factor = col2.number_input("Enter the scaling factor", value=recom_sf)
        direction = col2.radio(
            "Select text direction",
            ("Horizontal", "Vertical"))
    image = uploaded_file if uploaded_file else camera
    # processing starts here
    if option and (image or text):
        # run button
        run = col1.button("Run")
        if run:
            if image:
                stylized = stylize_face(image, option, device)
                if stylized is not None:
                    blended = blend_face(option, stylized)
                    if blended is not None: 
                        if text:
                            text = text.upper()
                            str_image = stylize_text(font, text, direction, device)
                            if str_image is not None:
                                final_image = insert_text(blended, str_image, cx, cy, scale_factor)
                                if final_image is not None:
                                    st.success("Your custom creative has been created!")
                                    st.image(final_image)
                        else:
                            st.success("Your custom creative has been created!")
                            st.image(blended)
            else:
                blended = cv2.imread(data[option]["path_textless"])[..., ::-1]
                text = text.upper()
                str_image = stylize_text(font, text, direction, device)
                if str_image is not None:
                    final_image = insert_text(blended, str_image, cx, cy, scale_factor)
                    if final_image is not None:
                        st.success("Your custom creative has been created!")
                        st.image(final_image)


if __name__ == "__main__":
    device = "cpu"
    main(device)