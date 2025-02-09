# Import necessary libraries and modules
from streamlit_drawable_canvas import st_canvas  # For enabling drawing on canvas in Streamlit
from ultralytics import YOLO  # For YOLO model integration
from PIL import Image  # For handling image processing
import streamlit as st  # For creating Streamlit web apps
import cv2, os, io, zipfile  # For image processing, file handling, and zip file creation
import numpy as np  # For numerical computations
import fitz  # PyMuPDF library for PDF processing


# Initialize YOLO model in Streamlit session state if not already loaded
# This ensures the model is loaded only once and shared across the session
if 'yolo_model' not in st.session_state:
    st.session_state.yolo_model = YOLO(r'./model/tile_detectionV2(1500).pt')


def pdf_to_images(upload_pdfs):
    """
        Convert a PDF file to a list of images, where each page of the PDF is represented as an image.

        This function saves the uploaded PDF file to a temporary location, converts each page into an
        image using PyMuPDF (fitz), and stores the images as NumPy arrays. After processing, the
        temporary file is deleted.

        Args:
            upload_pdfs (File): An uploaded file-like object representing the PDF to be converted.

        Returns:
            list: A list of NumPy arrays, where each array corresponds to an image of a PDF page.

        Raises:
            Exception: Logs errors encountered during PDF processing or image conversion.
            PermissionError: Logs errors if the temporary file cannot be deleted.

        Notes:
            - The function temporarily saves the uploaded PDF file to the current working directory.
            - Ensure that the uploaded PDF has a valid format and is accessible.
            - Adjust the DPI value in the `get_pixmap` call to control the resolution of the output images.
    """
    temp_file_path = os.path.join(os.getcwd(), upload_pdfs.name)

    # Write the uploaded PDF to a temporary file
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(upload_pdfs.read())

    images = []
    pdf_document = None
    try:
        # Open the PDF using PyMuPDF
        pdf_document = fitz.open(temp_file_path)
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            # pix = page.get_pixmap(dpi=50)  # Adjust DPI as needed
            pix = page.get_pixmap(dpi=200)  # Best Resolution For Me
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(np.array(img))  # Store as NumPy array
    except Exception as e:
        print(f"Error processing PDF {upload_pdfs.name}: {str(e)}")
    finally:
        # Ensure the PDF document is closed properly
        if pdf_document:
            pdf_document.close()

        # Ensure the temporary file is removed
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except PermissionError as e:
            print(f"Could not delete temporary file {temp_file_path}: {str(e)}")

    return images


def detect_tiles_using_ED(image):
    """
        Detect tiles in an image using a YOLO model and Edge Detection (ED).

        This function processes the input image to detect tiles based on bounding boxes predicted
        by a YOLO model. It filters bounding boxes based on confidence scores and area constraints.

        Args:
            image (PIL.Image.Image or np.ndarray): The input image for tile detection.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The original input image as a NumPy array.
                - list: A list of bounding boxes, where each bounding box is represented as
                  [x1, y1, x2, y2].

        Bounding Box Filtering:
            - Only includes boxes with a confidence score of 0.55 or higher.
            - Excludes boxes where the relative area is 40% or more of the total image area.

        Notes:
            - Ensure `st.session_state.yolo_model` is properly initialized with a YOLO model.
            - The input image is expected to have a shape of (width, height, channels).
            - The relative area of a bounding box is calculated as:
              ((box_width * box_height) / (image_width * image_height)) * 100
    """
    image = np.array(image)
    w, h, _ = image.shape

    results = st.session_state.yolo_model.predict(image)

    bounding_box = []
    if results[0].masks:
        for result in results:
            for mask, box in zip(result.masks.xy, result.boxes):
                confidence = box.conf.cpu().numpy()

                if confidence >= 0.55:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    if ((((x2 - x1) * (y2 - y1)) / (w * h)) * 100) < 40:
                        bounding_box.append([x1, y1, x2, y2])

    return image, bounding_box


def main():
    st.set_page_config('Tile Detection', page_icon='📰', layout='wide')

    # main_content, right_panel = st.columns([3, 1])
    main_content, right_panel = st.columns([5, 2])
    # main_content, right_panel = st.columns([6, 4])

    with main_content:
        st.title('Tile Detection')
        st.sidebar.title("Navigation")

        # PDF Upload
        pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
        if pdf_file:
            # Check if a new file is uploaded
            if 'last_uploaded_pdf_name' not in st.session_state or st.session_state.last_uploaded_pdf_name != pdf_file.name:
                st.session_state.images = None

                # For Copy and paste and remove part in show image.
                st.session_state.copy_annotation = None
                st.session_state.remove_annotation = False
                st.session_state.paste_status = False

                # For select pages annotation.
                st.session_state.select_box_cropped = False

                # For Annotation Model Status
                st.session_state.annotation_mode = False
                st.session_state.select_box = False
                st.session_state.annotations = []

                # For Index of the page in PDFs  and Name
                st.session_state.idx = 0
                st.session_state.last_uploaded_pdf_name = pdf_file.name

                # For Saving bounding box in dictionary.
                st.session_state.Cropped_image_Dic = {}

            # Convert PDF to images if not done already
            if 'images' not in st.session_state or st.session_state.images is None:
                with st.spinner("Converting PDF to images..."):
                    st.session_state.images = pdf_to_images(pdf_file)
                    st.session_state.Cropped_image_Dic = {idx: {'name': [],
                                                                'size': [],
                                                                'finish': [],
                                                                'bb': []} for idx in range(len(st.session_state.images))}

            idx = st.session_state.get('idx', 0)
            num_pages = len(st.session_state.images)



    # --------------------------------: Show Image and Add BB in Dict: Cropped_image_Dic : -------------------------------------------

            with (st.expander('Show Image')):
                detected_img, bounding_box = detect_tiles_using_ED(np.array(st.session_state.images[idx]))


                if st.session_state.Cropped_image_Dic[st.session_state.idx]['bb'] is not None and \
                        len(st.session_state.Cropped_image_Dic[st.session_state.idx]['bb']) == 0:

                    for bb in bounding_box:
                        if st.session_state.Cropped_image_Dic[st.session_state.idx]['bb'] is not None and \
                                bb not in st.session_state.Cropped_image_Dic[st.session_state.idx]['bb']:

                            st.session_state.Cropped_image_Dic[st.session_state.idx]['bb'].append(bb)

                    # Show Label For Each Box.
                    if len(st.session_state.Cropped_image_Dic[st.session_state.idx]['name']) < len(
                            st.session_state.Cropped_image_Dic[st.session_state.idx]['bb']):

                        st.session_state.Cropped_image_Dic[st.session_state.idx]['name'] = [f"Box {i + 1}" for i in range(
                            len(st.session_state.Cropped_image_Dic[st.session_state.idx]['bb']))]

                        st.session_state.Cropped_image_Dic[st.session_state.idx]['size'] = ["60x60" for _ in range(
                            len(st.session_state.Cropped_image_Dic[st.session_state.idx]['bb']))]

                        st.session_state.Cropped_image_Dic[st.session_state.idx]['finish'] = ['none' for _ in range(
                            len(st.session_state.Cropped_image_Dic[st.session_state.idx]['bb']))]

                if st.session_state.paste_status:
                    bounding_box = st.session_state.copy_annotation
                    st.session_state.Cropped_image_Dic[st.session_state.idx]['name'] = [f"Box {i + 1}" for i in range(
                        len(bounding_box))]

                    st.session_state.Cropped_image_Dic[st.session_state.idx]['size'] = ["60x60" for _ in range(
                        len(bounding_box))]

                    st.session_state.Cropped_image_Dic[st.session_state.idx]['finish'] = ['none' for _ in range(
                        len(bounding_box))]
                    st.session_state.paste_status = False
                else: pass

                if st.session_state.remove_annotation:
                    st.session_state.Cropped_image_Dic[st.session_state.idx]['name'] = None
                    st.session_state.Cropped_image_Dic[st.session_state.idx]['bb'] = None
                    st.session_state.Cropped_image_Dic[st.session_state.idx]['size'] = None
                    st.session_state.Cropped_image_Dic[st.session_state.idx]['finish'] = None
                    st.session_state.remove_annotation = False
                else: pass



                if st.session_state.Cropped_image_Dic[st.session_state.idx]['bb'] is None:
                    bounding_box = []

                elif len(st.session_state.Cropped_image_Dic[st.session_state.idx]['bb']) > 0:
                    for i, [x, y, w, h] in enumerate(st.session_state.Cropped_image_Dic[st.session_state.idx]['bb']):
                        cv2.rectangle(detected_img, (x, y), (w, h), (0, 255, 0), 2)

                        # Print Label Above Tiles
                        cv2.putText(detected_img, st.session_state.Cropped_image_Dic[st.session_state.idx]['name'][i],
                                    (x, y+15), cv2.FONT_ITALIC, 0.55, (255, 0, 0), 2)


                else: pass


                # st.info(st.session_state.Cropped_image_Dic[st.session_state.idx]['bb'])
                # st.info(st.session_state.Cropped_image_Dic[st.session_state.idx]['name'])
                # st.info(st.session_state.Cropped_image_Dic[st.session_state.idx]['size'])
                # st.info(st.session_state.Cropped_image_Dic[st.session_state.idx]['finish'])

                if not st.session_state.annotation_mode:
                    # st.image(detected_img, caption=f"Detected Page {idx + 1}", width=750)
                    # st.image(detected_img, caption=f"Detected Page {idx + 1}", width=750, use_column_width='auto')
                    st.image(detected_img, caption=f"Detected Page {idx + 1}", width=750, use_column_width='always')

    # --------------------------------: Manual Annotation :--------------------------------------------------------

                if st.session_state.annotation_mode:
                    pil_image = Image.fromarray(np.array(st.session_state.images[st.session_state.idx]))
                    realtime_update = st.sidebar.checkbox("Update in realtime", True)
                    st.info((pil_image.height, pil_image.width))

                    # Create a canvas component
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",
                        update_streamlit=realtime_update,
                        stroke_width=2,
                        # height=pil_image.height,
                        # width=pil_image.width,
                        height=600,
                        width=800,
                        background_image=pil_image,
                        drawing_mode="rect",
                        key="canvas",
                    )

                    if canvas_result.json_data is not None:
                        if len(canvas_result.json_data["objects"]) > 0:
                            st.session_state.annotations.append(canvas_result.json_data["objects"])

                # Buttons to end annotation mode and apply annotations
                if st.session_state.annotation_mode:
                    col1, col2, col3 = st.columns(3)

                    # Button to apply to current page only
                    if col1.button("Apply to Current Page"):
                        if len(st.session_state.annotations) > 0:
                            for i, annotation in enumerate(st.session_state.annotations[-1]):
                                x = annotation['left']
                                y = annotation['top']
                                w = annotation['width']
                                h = annotation['height']
                                current_page = st.session_state.idx

                                st.info(st.session_state.Cropped_image_Dic[current_page]['bb'])
                                if st.session_state.Cropped_image_Dic[current_page]['bb'] is None:
                                    st.session_state.Cropped_image_Dic[current_page]['bb'] = [
                                        [x, y, w + x, h + y]]
                                    st.session_state.Cropped_image_Dic[current_page]['name'] = [
                                        f"Box {i + 1}"]
                                    st.session_state.Cropped_image_Dic[current_page]['size'] = ["60x60"]
                                    st.session_state.Cropped_image_Dic[current_page]['finish'] = ['none']

                                elif [x, y, w + x, h + y] not in \
                                        st.session_state.Cropped_image_Dic[current_page]['bb']:
                                    st.session_state.Cropped_image_Dic[current_page]['bb'].append(
                                        [x, y, w + x, h + y])
                                    st.session_state.Cropped_image_Dic[current_page]['name'].append(
                                        f"Box {len(st.session_state.Cropped_image_Dic[current_page]['name']) + i + 1}")
                                    st.session_state.Cropped_image_Dic[current_page]['size'].append("60x60")
                                    st.session_state.Cropped_image_Dic[current_page]['finish'].append('none')

                                else:
                                    pass

                            st.session_state.annotation_mode = False
                            st.session_state.annotations = []
                            st.success(f"Annotation applied to page {current_page + 1}")
                            st.experimental_rerun()

                    # Button to apply to all pages
                    if col2.button("Apply to All Pages"):
                        if len(st.session_state.annotations) > 0:
                            for i, annotation in enumerate(st.session_state.annotations[-1]):
                                x = annotation['left']
                                y = annotation['top']
                                w = annotation['width']
                                h = annotation['height']

                                # Apply to all pages
                                for page_idx in st.session_state.Cropped_image_Dic.keys():
                                    if st.session_state.Cropped_image_Dic[page_idx]['bb'] is None:
                                        st.session_state.Cropped_image_Dic[page_idx]['bb'] = [[x, y, w + x, h + y]]
                                        st.session_state.Cropped_image_Dic[page_idx]['name'] = [f"Box {i + 1}"]
                                        st.session_state.Cropped_image_Dic[page_idx]['size'] = ["60x60"]
                                        st.session_state.Cropped_image_Dic[page_idx]['finish'] = ['none']

                                    elif [x, y, w + x, h + y] not in \
                                            st.session_state.Cropped_image_Dic[page_idx]['bb']:
                                        st.session_state.Cropped_image_Dic[page_idx]['bb'].append([x, y, w + x, h + y])
                                        st.session_state.Cropped_image_Dic[page_idx]['name'].append(
                                            f"Box {len(st.session_state.Cropped_image_Dic[page_idx]['name']) + i + 1}")
                                        st.session_state.Cropped_image_Dic[page_idx]['size'].append("60x60")
                                        st.session_state.Cropped_image_Dic[current_page]['finish'].append('none')

                                    else:
                                        pass

                            st.session_state.annotation_mode = False
                            st.session_state.annotations = []
                            st.success("Annotation applied to all pages")
                            st.experimental_rerun()

                    # New Button to Select Pages
                    if col3.button("Select Pages"):
                        st.session_state.select_box = True

                    if st.session_state.select_box:
                        selected_pages = st.multiselect(
                            "Select pages to apply annotations:",
                            options=list(range(1, len(st.session_state.images) + 1)),  # Page numbers
                            default=[st.session_state.idx + 1]  # Default to current page
                        )

                    if st.session_state.select_box:
                        cancel_selected, apply_selected = st.columns(2)
                        if apply_selected.button('Apply Annotations'):
                            if len(st.session_state.annotations) > 0:
                                for i, annotation in enumerate(st.session_state.annotations[-1]):
                                    x = annotation['left']
                                    y = annotation['top']
                                    w = annotation['width']
                                    h = annotation['height']

                                    # Apply to selected pages
                                    for page in selected_pages:
                                        page_idx = page - 1  # Convert to zero-based index
                                        if st.session_state.Cropped_image_Dic[page_idx]['bb'] is None:
                                            st.session_state.Cropped_image_Dic[page_idx]['bb'] = [
                                                [x, y, w + x, h + y]]
                                            st.session_state.Cropped_image_Dic[page_idx]['name'] = [f"Box {i + 1}"]
                                            st.session_state.Cropped_image_Dic[page_idx]['size'] = ["60x60"]
                                            st.session_state.Cropped_image_Dic[page_idx]['finish'] = ['none']

                                        elif [x, y, w + x, h + y] not in \
                                                st.session_state.Cropped_image_Dic[page_idx]['bb']:
                                            st.session_state.Cropped_image_Dic[page_idx]['bb'].append([x, y, w + x, h + y])
                                            st.session_state.Cropped_image_Dic[page_idx]['name'].append(
                                                f"Box {len(st.session_state.Cropped_image_Dic[page_idx]['name']) + i + 1}")
                                            st.session_state.Cropped_image_Dic[page_idx]['size'].append("60x60")
                                            st.session_state.Cropped_image_Dic[current_page]['finish'].append('none')

                                        else:
                                            pass

                                st.session_state.annotation_mode = False
                                st.session_state.select_box = False
                                st.session_state.annotations = []
                                st.success(
                                    f"Annotation applied to pages: {', '.join(map(str, selected_pages))}")
                                st.experimental_rerun()

                        if cancel_selected.button("Cancel Selected"):
                            # st.session_state.annotation_mode = False
                            st.session_state.select_box = False
                            st.session_state.annotations = []
                            st.success(f"Annotation applied to pages: {', '.join(map(str, selected_pages))}")
                            st.experimental_rerun()



    # --------------------------------: Remove, Copy and Edit In Annotation : -------------------------------------------

                remove_col, col_copy, col_paste = st.columns(3)

                # For Removing All Annotation From Image.
                with remove_col:
                    if st.button('Remove Annotation'):
                        if st.session_state.Cropped_image_Dic[st.session_state.idx]['bb'] is not None:
                            st.session_state.Cropped_image_Dic[st.session_state.idx]['bb'] = []
                            st.session_state.remove_annotation = True
                            st.experimental_rerun()

                with col_copy:
                    if st.button('Copy Annotation'):
                        st.session_state.copy_annotation = bounding_box
                        st.session_state.Cropped_image_Dic[st.session_state.idx]['bb'] = st.session_state.copy_annotation

                with col_paste:
                    if st.button('Paste Annotation'):
                        st.session_state.Cropped_image_Dic[st.session_state.idx]['bb'] = st.session_state.copy_annotation
                        st.session_state.paste_status = True
                        st.experimental_rerun()

                # Button to start annotation mode
                start_annotation, cancel_annotation, _ = st.columns(3)
                if start_annotation.button("Manual Annotation"):
                    st.session_state.annotation_mode = True
                    st.experimental_rerun()

                # Button to cancel annotation
                if st.session_state.annotation_mode:
                    if cancel_annotation.button("Cancel Annotation"):
                        st.session_state.annotation_mode = False
                        st.session_state.annotations = []
                        st.warning("Annotation cancelled")
                        st.experimental_rerun()

    # -----------------------------------: Next & Previous Pages Section:--------------------------------------------------

            # Page Navigation
            st.sidebar.subheader('Pages')
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button('Previous', disabled=(idx <= 0)):
                    if idx > 0:
                        st.session_state.idx -= 1
                        st.experimental_rerun()

            with col2:
                if st.button('Next', disabled=(idx >= num_pages - 1)):
                    if idx < num_pages - 1:
                        st.session_state.idx += 1
                        st.experimental_rerun()

    # --------------------------------: Go Pages sidebar :---------------------------------------------------------------

            # Page Navigation Buttons in Sidebar
            with st.sidebar.expander("Go to Page", expanded=False):
                num_columns = 4  # Number of buttons per row
                cols = st.columns(num_columns)

                for i in range(num_pages):
                    col_index = i % num_columns  # Determine the column index
                    with cols[col_index]:
                        if st.button(f"{i + 1}"):
                            st.session_state.idx = i
                            st.experimental_rerun()

    # --------------------------------: Export Options For Download Crop Image :---------------------------------------

            with st.expander("Export Options For Download Crop Image"):
                download_1_Image, download_specific_Image, download_all_Image = st.columns(3)
                with download_1_Image:
                    if st.button("Export 1 Page"):
                        cropped_data = st.session_state.Cropped_image_Dic[st.session_state.idx]
                        if not cropped_data['bb']:
                            st.error("This Page Doesn't Have Tiles")
                        else:
                            st.success(f"Success! Cropped {len(cropped_data['bb'])} Images.")
                            zip_buffer = io.BytesIO()

                            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                                for i, ([x, y, w, h], size, finish) in enumerate(zip(cropped_data['bb'],
                                                                                     cropped_data['size'],
                                                                                     cropped_data['finish'])):

                                    cropped_img = np.array(st.session_state.images[st.session_state.idx])[y:h, x:w]
                                    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                                    _, img_bytes = cv2.imencode('.png', cropped_img)

                                    folder_path = os.path.join(str(size), finish)
                                    file_path = os.path.join(folder_path, f"cropped_image_{i + 1}.png")

                                    zf.writestr(file_path, img_bytes.tobytes())


                            zip_buffer.seek(0)
                            st.download_button(
                                label="Download Cropped Images as Zip",
                                data=zip_buffer,
                                file_name=f"{pdf_file.name}.zip",
                                mime="application/zip"
                            )
                            st.success("Cropped images saved and ready for download!")

                with download_specific_Image:
                    if st.button("Export Specific Pages"):
                        st.session_state.select_box_cropped = True

                    if st.session_state.select_box_cropped:
                        selected_pages = st.multiselect(
                            "Select pages to apply annotations:",
                            options=list(range(1, len(st.session_state.images) + 1))  # Page numbers
                        )

                    if st.session_state.select_box_cropped and st.button('Finish Selected'):
                        st.info(selected_pages)
                        for i in selected_pages:
                            i -= 1
                            if st.session_state.Cropped_image_Dic[i]['bb'] is None:
                                continue

                            elif len(st.session_state.Cropped_image_Dic[i]['bb']) == 0:
                                detected_img, bounding_box = detect_tiles_using_ED(np.array(st.session_state.images[i]))
                                for j, bb in enumerate(bounding_box):
                                    if st.session_state.Cropped_image_Dic[i]['bb'] is not None and \
                                            bb not in st.session_state.Cropped_image_Dic[i]['bb']:

                                        st.session_state.Cropped_image_Dic[i]['bb'].append(bb)
                                        st.session_state.Cropped_image_Dic[i]['name'].append(f"Box {j+1}")
                                        st.session_state.Cropped_image_Dic[i]['size'].append("60x60")
                                        st.session_state.Cropped_image_Dic[i]['finish'].append('none')

                            else: pass

                        st.success(
                            f'Success Crop Images {sum([len(st.session_state.Cropped_image_Dic[i-1]["bb"]) for i in selected_pages if st.session_state.Cropped_image_Dic[i - 1]["bb"] is not None])}')
                        # Create an in-memory zip file
                        zip_buffer = io.BytesIO()


                        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                            for idx_page in selected_pages:
                                idx_page -= 1
                                page_data = st.session_state.Cropped_image_Dic[idx_page]
                                if page_data['bb'] is not None:
                                    for i, ([x, y, w, h], size, finish) in enumerate(
                                            zip(page_data['bb'], page_data['size'], page_data['finish'])):

                                        cropped_img = np.array(st.session_state.images[idx_page])[y:h, x:w]
                                        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                                        _, img_bytes = cv2.imencode('.png', cropped_img)

                                        folder_path = os.path.join(str(size), finish)
                                        file_path = os.path.join(folder_path, f"cropped_image_{idx_page + 1}_{i + 1}.png")


                                        zf.writestr(file_path, img_bytes.tobytes())

                        zip_buffer.seek(0)
                        st.download_button(
                            label="Download Cropped Images as Zip",
                            data=zip_buffer,
                            file_name=f"{pdf_file.name}.zip",
                            mime="application/zip"
                        )
                        st.success("Cropped images saved and ready for download!")
                        st.session_state.select_box_cropped = False



                with download_all_Image:
                    if st.button("Export All Tiles"):

                        for i in st.session_state.Cropped_image_Dic.keys():
                            if st.session_state.Cropped_image_Dic[i]['bb'] is None:
                                continue

                            elif len(st.session_state.Cropped_image_Dic[i]['bb']) == 0:
                                detected_img, bounding_box = detect_tiles_using_ED(np.array(st.session_state.images[i]))
                                for j, bb in enumerate(bounding_box):
                                    if st.session_state.Cropped_image_Dic[i]['bb'] is not None and bb not in st.session_state.Cropped_image_Dic[i]['bb']:
                                        st.session_state.Cropped_image_Dic[i]['bb'].append(bb)
                                        st.session_state.Cropped_image_Dic[i]['name'].append(f"Box {j + 1}")
                                        st.session_state.Cropped_image_Dic[i]['size'].append("60x60")
                                        st.session_state.Cropped_image_Dic[i]['finish'].append('none')

                            else: pass

                        total_cropped_images = sum(
                            len(st.session_state.Cropped_image_Dic[i]["bb"]) for i in
                            st.session_state.Cropped_image_Dic.keys()
                            if st.session_state.Cropped_image_Dic[i]["bb"] is not None
                        )
                        st.success(f"Successfully cropped {total_cropped_images} images.")

                        # Create an in-memory zip file
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                            for idx_page in range(num_pages):
                                cropped_data = st.session_state.Cropped_image_Dic[idx_page]

                                if cropped_data['bb'] is not None:
                                    for i, ([x, y, w, h], size, finish) in enumerate(
                                            zip(cropped_data['bb'], cropped_data['size'], cropped_data['finish'])):

                                        cropped_img = np.array(st.session_state.images[idx_page])[y:h, x:w]
                                        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                                        _, img_bytes = cv2.imencode('.png', cropped_img)

                                        folder_path = os.path.join(str(size), finish)
                                        file_path = os.path.join(folder_path, f"cropped_image_{idx_page + 1}_{i + 1}.png")

                                        zf.writestr(file_path, img_bytes.tobytes())

                        zip_buffer.seek(0)
                        st.download_button(
                            label="Download Cropped Images as Zip",
                            data=zip_buffer,
                            file_name=f"{pdf_file.name}.zip",
                            mime="application/zip"
                        )
                        st.success("Cropped images saved and ready for download!")

    # Right panel (simulating a sidebar)
    with right_panel:

        # Page Navigation
        if 'Cropped_image_Dic' in st.session_state:
            # Get current page index
            current_page = st.session_state.get('idx', 0)

            # Display metadata for the current page
            st.subheader(f"Page {current_page + 1} Metadata")


            if st.session_state.Cropped_image_Dic[current_page]['name'] is not None and \
                    st.session_state.Cropped_image_Dic[current_page]['bb'] is not None:

                # Input names for each bounding box
                for i, _ in enumerate(st.session_state.Cropped_image_Dic[current_page]['bb']):

                    # Box_Name, Standard_size, Finish_Size, Box_remove = st.columns(4)

                    Box_Name, Box_remove = st.columns(2)
                    Standard_size, Finish_Size = st.columns(2)

                    # Input field for box name
                    with Box_Name:
                        box_name = st.text_input(
                            f"Name: Box {i + 1}",
                            value=st.session_state.Cropped_image_Dic[current_page]['name'][i],
                            key=f"box_name_{current_page}_{i}"
                        )

                        # Update the name in the dictionary
                        st.session_state.Cropped_image_Dic[current_page]['name'][i] = box_name

                    with Standard_size:
                        # Input field for Size
                        standard_size = st.selectbox(
                            label=f"Standard Sizes",
                            options=[st.session_state.Cropped_image_Dic[current_page]['size'][i],
                                     "120x120",
                                     "60x60",
                                     'other'],
                            key=f"box_size_{current_page}_{i}"
                        )

                        if standard_size == 'other':
                            custom_size = st.text_input('Size', key=f"custom_size_{current_page}_{i}")
                            if custom_size:
                                st.session_state.Cropped_image_Dic[current_page]['size'][i] = custom_size

                        # Update the Size in the dictionary
                        else: st.session_state.Cropped_image_Dic[current_page]['size'][i] = standard_size



                    with Finish_Size:
                        # Input field for Size
                        finish_size_select = st.selectbox(
                            label=f"Finsh Type",
                            options=[st.session_state.Cropped_image_Dic[current_page]['finish'][i],
                                     'Matte', 'Honed', 'Textured&Rustic', 'other'],
                            key=f"box_finish_{current_page}_{i}"
                        )

                        if finish_size_select == 'other':
                            finish_size = st.text_input('finish', key=f"finish_size_{current_page}_{i}")
                            if finish_size:
                                st.session_state.Cropped_image_Dic[current_page]['finish'][i] = finish_size

                        else: st.session_state.Cropped_image_Dic[current_page]['finish'][i] = finish_size_select

                    with Box_remove:
                        st.markdown(
                            f"<p style='font-size: 16px; margin-bottom: 0px;'>Remove Box</p>",
                            unsafe_allow_html=True
                        )
                        if st.button(f"Remove ", key=f"Remove_box_{i}"):
                            st.session_state.Cropped_image_Dic[current_page]['name'].remove(st.session_state.Cropped_image_Dic[current_page]['name'][i])
                            st.session_state.Cropped_image_Dic[current_page]['bb'].remove(st.session_state.Cropped_image_Dic[current_page]['bb'][i])
                            st.session_state.Cropped_image_Dic[current_page]['size'].remove(st.session_state.Cropped_image_Dic[current_page]['size'][i])
                            st.experimental_rerun()

                    st.markdown('---')


                if st.button('Confirm Updates'):
                    st.success('Metadata Update')
                    st.experimental_rerun()


                # Additional page information
                st.divider()
                st.write("**Page Details:**")
                st.text(f"Total Pages: {len(st.session_state.images)}")
                st.text(f"Current Page: {current_page + 1}")


if __name__ == "__main__":
    main()