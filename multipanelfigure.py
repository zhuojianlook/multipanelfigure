import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

# Function to create a multi-panel figure with the given template and customization options
def create_multi_panel_figure(template, images, v_label_customizations, spacing, panel_label_customization):
    rows, cols = map(int, template.split('x'))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=300)

    # Sort images based on specified position
    images.sort(key=lambda x: x['position'])

    # Apply spacing adjustments for the subplots
    plt.subplots_adjust(wspace=spacing, hspace=spacing)

    # Normalize axes array to 2D if rows or cols is 1
    if rows == 1 and cols != 1:
        axes = np.array([axes])  # Convert to 2D array for consistency
    elif cols == 1 and rows != 1:
        axes = np.array([[ax] for ax in axes])  # Convert to 2D array for consistency

    # Iterate over the grid and create each panel
    for i, (img_data, label_custom) in enumerate(zip(images, panel_label_customization)):
        row, col = divmod(i, cols)
        ax = axes[row, col] if isinstance(axes, np.ndarray) else axes[i]
    
        # Convert byte data back to image for display
        image = Image.open(io.BytesIO(img_data['bytes'])).convert('RGB')
    
        # The crop and display should align here
        ax.imshow(image)
        ax.axis('off')
    
        # Adjust label position to increase the border
        label_x = label_custom['position_dict']['x']
        label_y = label_custom['position_dict']['y']
        border_offset = 0.05  # Adjust this value to increase or decrease the border

        if 'upper' in label_custom['position_dict']['loc']:
            label_y -= border_offset
        else:
            label_y += border_offset

        if 'right' in label_custom['position_dict']['loc']:
            label_x -= border_offset
        else:
            label_x += border_offset

        # Add panel labels with customization
        ax.text(
            x=label_x,
            y=label_y,
            s=label_custom['text'],
            color=label_custom['color'],
            fontsize=label_custom['font_size'],
            fontstyle=label_custom['style'],
            fontname=label_custom['font_name'],
            weight='bold',
            verticalalignment='top' if 'upper' in label_custom['position_dict']['loc'] else 'bottom',
            horizontalalignment='right' if 'right' in label_custom['position_dict']['loc'] else 'left',
            transform=ax.transAxes
        )


    # Add horizontal labels with individual customization
    if rows == 1:
        for ax, h_label_custom in zip(axes.flatten(), horizontal_label_customizations):
            ax.annotate(
                h_label_custom['text'],
                xy=(0.5, 1 + h_label_custom['distance']),
                xycoords='axes fraction',
                fontsize=h_label_custom['size'],
                color=h_label_custom['color'],
                fontname=h_label_custom['font'],
                fontstyle=h_label_custom['style'],
                rotation=0,
                verticalalignment='bottom',
                horizontalalignment='center'
            )
    else:
        for ax, h_label_custom in zip(axes[0], horizontal_label_customizations):
            ax.annotate(
                h_label_custom['text'],
                xy=(0.5, 1 + h_label_custom['distance']),
                xycoords='axes fraction',
                fontsize=h_label_custom['size'],
                color=h_label_custom['color'],
                fontname=h_label_custom['font'],
                fontstyle=h_label_custom['style'],
                rotation=0,
                verticalalignment='bottom',
                horizontalalignment='center'
            )

    # Add vertical labels with individual customization
    if cols == 1:
        for ax, v_label_custom in zip(axes.flatten(), v_label_customizations):
            ax.annotate(
                v_label_custom['text'],
                xy=(-v_label_custom['distance'], 0.5),
                xycoords='axes fraction',
                fontsize=v_label_custom['size'],
                color=v_label_custom['color'],
                fontname=v_label_custom['font'],
                fontstyle=v_label_custom['style'],
                rotation=90,
                verticalalignment='center',
                horizontalalignment='right'
            )
    else:
        for ax, v_label_custom in zip(axes[:, 0], v_label_customizations):
            ax.annotate(
                v_label_custom['text'],
                xy=(-v_label_custom['distance'], 0.5),
                xycoords='axes fraction',
                fontsize=v_label_custom['size'],
                color=v_label_custom['color'],
                fontname=v_label_custom['font'],
                fontstyle=v_label_custom['style'],
                rotation=90,
                verticalalignment='center',
                horizontalalignment='right'
            )

    # Save the figure to a BytesIO stream and rewind to the beginning
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)



    # Close the figure to avoid memory issues
    plt.close(fig)

    return buf


def create_blank_grid(template):
    rows, cols = map(int, template.split('x'))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=300)

    for i in range(rows * cols):
        row, col = divmod(i, cols)
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')
        ax.text(0.5, 0.5, f'Position {i}', ha='center', va='center')

    return fig


# Convert the buffer to a data URL for download link
def get_image_download_link(img_buffer, filename, text):
    b64 = base64.b64encode(img_buffer.getvalue()).decode()
    href = f'<a href="data:image/tiff;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Streamlit UI for input and customization
st.title('Multi-Panel Figure Generator')
st.caption('For TIF and TIFF files only, specify your grid size')
# Streamlit UI for input and customization in the sidebar
st.sidebar.title('Reference Grid')

# Template input in the sidebar
template = st.sidebar.text_input('Template (e.g., "3x3")', '3x3')
rows, cols = map(int, template.split('x'))

# Spacing between panels in the sidebar
#spacing = st.sidebar.slider('Does not do anything', 0.0, 0.1, 0.02, key='spacing_slider')


# Button to display a blank grid with positions in the sidebar
if st.sidebar.button('Show Grid Positions'):
    fig = create_blank_grid(template)
    st.sidebar.pyplot(fig)


# Spacing between panels
spacing = st.slider('Spacing Between Panels', 0.0, 0.1, 0.02)

# File uploader for images
uploaded_files = st.file_uploader('Upload .tif .jpg .png files', type=['tif', 'png', 'jpg'], accept_multiple_files=True)

# Initialize lists for label customizations and image data
panel_label_customization = []
v_label_customizations = []
image_data = []

if uploaded_files and len(uploaded_files) == rows * cols:
    # Image cropping and panel label customization
    for i, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name  # Get the filename of the uploaded file
        st.write(f"{file_name} - Cropped")  # Display the filename instead of 'Image1-a'
        image = Image.open(uploaded_file)
        width, height = image.size

        # Ensure the crop is a square and calculate the maximum offset based on image dimensions
        side_length = min(width, height)
        max_offset = (width - side_length) // 2

        # Slider for crop offset
        crop_offset = st.slider(f"Crop Offset for Image {i+1}", -max_offset, max_offset, 0, key=f"crop_offset_{i}")

        # Calculate left and right coordinates for cropping, keeping the square dimensions
        left = max_offset + crop_offset
        right = width - max_offset + crop_offset

        # Crop the image and display
        cropped_image = image.crop((left, 0, right, height))
        st.image(cropped_image, caption=f"Cropped Image {i+1}")

        # Convert the cropped image to bytes and store cropping data
        img_bytes = io.BytesIO()
        cropped_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        # Input for specifying position in the grid
        position = st.number_input(f'Position for Image {i+1} (0-{rows*cols-1})', min_value=0, max_value=rows*cols-1, key=f"pos_{i}")
        
        # When appending the data, use the file name without the extension as the label text
        file_label = file_name.rsplit('.', 1)[0]  # Remove the file extension from the label
        
        # Append image data including position and cropping details
        image_data.append({
            'bytes': img_bytes.getvalue(),
            'crop': (left, 0, right, height),
            'position': position,
            'label': file_label  # Store the file label
        })

        # Panel label customization for each image
        with st.container():
            text = st.text_input(f'Label text for Image {i+1}', f'{chr(97+i)}')
            position_option = st.selectbox(f'Label position for Image {i+1}', ['upper left', 'upper right', 'lower left', 'lower right'], index=1)
            font_size = st.number_input(f'Font size for Image {i+1}', value=10)
            color = st.color_picker(f'Label color for Image {i+1}', '#000000')
            font_name = st.selectbox(f'Font for Image {i+1}', ['Arial', 'Calibri', 'Times New Roman', 'Sans-serif'])
            style = st.selectbox(f'Font style for Image {i+1}', ['normal', 'italic', 'oblique'])

            # Define position dict based on the selected option
            position_dict = {
                'upper left': {'loc': 'upper left', 'x': 0, 'y': 1},
                'upper right': {'loc': 'upper right', 'x': 1, 'y': 1},
                'lower left': {'loc': 'lower left', 'x': 0, 'y': 0},
                'lower right': {'loc': 'lower right', 'x': 1, 'y': 0}
            }[position_option]

            # Add customization to the list
            panel_label_customization.append({
                'text': text,
                'position_dict': position_dict,
                'font_size': font_size,
                'color': color,
                'font_name': font_name,
                'style': style
            })



    # Customization for vertical figure labels
    for i in range(rows):
        with st.container():
            text = st.text_input(f'Label for Row {i+1}', f'Row {i+1}')
            size = st.number_input(f'Font Size for Row {i+1}', value=10)
            color = st.color_picker(f'Label Color for Row {i+1}', '#000000')
            font = st.selectbox(f'Font for Row {i+1}', ['Arial', 'Calibri', 'Times New Roman', 'Sans-serif'])
            style = st.selectbox(f'Font Style for Row {i+1}', ['normal', 'italic', 'oblique'])
            distance = st.slider(f'Distance from figure for Row {i+1}', 0.0, 1.0, 0.025)
            v_label_customizations.append({
                'text': text,
                'size': size,
                'color': color,
                'font': font,
                'style': style,
                'distance': distance
            })


	# Customization for horizontal figure labels
    horizontal_label_customizations = []
    for j in range(cols):
	    with st.container():
	        h_text = st.text_input(f'Label for Column {j+1}', f'Column {j+1}')
	        h_size = st.number_input(f'Font Size for Column {j+1}', value=10)
	        h_color = st.color_picker(f'Label Color for Column {j+1}', '#000000')
	        h_font = st.selectbox(f'Font for Column {j+1}', ['Arial', 'Calibri', 'Times New Roman', 'Sans-serif'])
	        h_style = st.selectbox(f'Font Style for Column {j+1}', ['normal', 'italic', 'oblique'])
	        h_distance = st.slider(f'Distance from figure for Column {j+1}', 0.0, 1.0, 0.025)
	        horizontal_label_customizations.append({
	            'text': h_text,
	            'size': h_size,
	            'color': h_color,
	            'font': h_font,
	            'style': h_style,
	            'distance': h_distance
	        })

    # Generate figure button
    if st.button('Generate Figure'):
        buf = create_multi_panel_figure(template, image_data, v_label_customizations, spacing, panel_label_customization)
        st.image(Image.open(buf), use_column_width=True)
        st.markdown(get_image_download_link(buf, 'multi_panel_figure.tif', 'Download Figure as TIF'), unsafe_allow_html=True)
else:
    st.error('Please upload the correct number of images as specified by the template.')


