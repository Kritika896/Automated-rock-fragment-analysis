import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import os
import io
import matplotlib.pyplot as plt


# ---- Utility Functions ----

def calculate_roundness(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter != 0:
        return (4 * np.pi * area) / (perimeter ** 2)
    return None

def convert_to_cm2(area_px, pixels_per_cm):
    return area_px / (pixels_per_cm ** 2)

def convert_to_mm2(area_px, pixels_per_mm):
    return area_px / (pixels_per_mm ** 2)

def calculate_average_diameter(contour):
    _, radius = cv2.minEnclosingCircle(contour)
    return 2 * radius

def calculate_volume(contour, roundness, width_mm, height_mm):
    if roundness >= 0.7:
        radius_mm = width_mm / 2
        return (4/3) * np.pi * (radius_mm ** 3)
    else:
        a = width_mm
        b = height_mm / 2
        return (4/3) * np.pi * a * (b ** 2)

def calculate_mass(volume_mm3, density_g_per_cm3):
    return volume_mm3 * (density_g_per_cm3 / 1000)

def calculate_height_width(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w, h

def detect_and_measure_contours(image, min_area_threshold, pixels_per_cm):
    results = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        roundness = calculate_roundness(contour)
        area_px = cv2.contourArea(contour)
        area_cm2 = convert_to_cm2(area_px, pixels_per_cm)

        if roundness and area_cm2 and area_px >= min_area_threshold:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            width_cm = w / pixels_per_cm
            height_cm = h / pixels_per_cm
            results.append({
                "Roundness": roundness,
                "Area (cm²)": area_cm2,
                "Width (cm)": width_cm,
                "Height (cm)": height_cm
            })
    return image, results

def crop_image_streamlit(img):
    height, width = img.shape[:2]
    
    st.subheader("Crop Image")
    x_start = st.slider("Start X", 0, width - 1, 0)
    x_end = st.slider("End X", x_start + 1, width, width)
    y_start = st.slider("Start Y", 0, height - 1, 0)
    y_end = st.slider("End Y", y_start + 1, height, height)

    cropped_img = img[y_start:y_end, x_start:x_end]
    
    st.image(cropped_img, caption="Cropped Image", channels="BGR", use_column_width=True)
    
    return cropped_img

def run_calibration():
    st.subheader("📐 Image Calibration using Homography")

    uploaded_file = st.file_uploader("Upload image for calibration", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("❌ Error: Unable to load image. Please check the file.")
            return

        try:
            calibrated_img = perform_homography(img)
            st.image(calibrated_img, caption="Calibrated Image", channels="BGR")

            # Save and download
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                cv2.imwrite(tmp_file.name, calibrated_img)
                with open(tmp_file.name, "rb") as file:
                    st.download_button(label="📥 Download Calibrated Image",
                                       data=file,
                                       file_name="calibrated_image.jpg",
                                       mime="image/jpeg")
        except Exception as e:
            st.error(f"❌ Error during homography: {str(e)}")

def perform_homography(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        src_points = np.array([point[0] for point in approx], dtype=np.float32)
    else:
        raise Exception("The largest contour is not a quadrilateral.")

    width, height = 1000, 700
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    H, _ = cv2.findHomography(src_points, dst_points)
    img_warped = cv2.warpPerspective(img, H, (width, height))

    # Optional saving logic (not used in streamlit flow but kept for desktop use)
    desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
    output_path = os.path.join(desktop_path, 'calibrated_image.jpg')
    if os.path.isdir(desktop_path) and cv2.imwrite(output_path, img_warped):
        print(f"Calibrated image successfully saved to: {output_path}")
    else:
        print("Failed to save the calibrated image.")

    return img_warped

def extract_contour_data(image, pixels_per_mm, density_g_per_cm3):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_details = []

    for color, contours in zip(["Green", "Red"], [contours_green, contours_red]):
        for contour_index, contour in enumerate(contours):
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            roundness = calculate_roundness(contour)
            if roundness is not None:
                area_mm2 = convert_to_mm2(area, pixels_per_mm)
                perimeter_mm = perimeter / pixels_per_mm
                average_diameter = calculate_average_diameter(contour) / pixels_per_mm
                width_px, height_px = calculate_height_width(contour)
                width_mm = width_px / pixels_per_mm
                height_mm = height_px / pixels_per_mm
                volume_mm3 = calculate_volume(contour, roundness, width_mm, height_mm)
                mass_g = calculate_mass(volume_mm3, density_g_per_cm3)
                contour_details.append({
                    "Color": color,
                    "Contour Index": contour_index + 1,
                    "Perimeter (mm)": perimeter_mm,
                    "Area (mm^2)": area_mm2,
                    "Roundness": roundness,
                    "Average Diameter (mm)": average_diameter,
                    "Width (mm)": width_mm,
                    "Height (mm)": height_mm,
                    "Volume (mm^3)": volume_mm3,
                    "Mass (g)": mass_g
                })

    return contour_details

def generate_gradation_curve(df):
    if 'Average Diameter (mm)' not in df.columns or 'Mass (g)' not in df.columns:
        st.error("❌ The required columns are not present in the CSV file")
        return

    min_diameter = df['Average Diameter (mm)'].min()
    max_diameter = df['Average Diameter (mm)'].max()
    bin_edges = np.logspace(np.log10(min_diameter), np.log10(max_diameter), num=21)

    df['Sieve Size (mm)'] = pd.cut(df['Average Diameter (mm)'], bins=bin_edges, labels=bin_edges[:-1], include_lowest=True)
    total_mass = df['Mass (g)'].sum()

    sieve_summary = df.groupby('Sieve Size (mm)').agg({'Mass (g)': 'sum'}).sort_index()
    sieve_summary['Cumulative Mass (g)'] = sieve_summary['Mass (g)'].cumsum()
    sieve_summary['% Passing'] = sieve_summary['Cumulative Mass (g)'] / total_mass * 100
    sieve_summary['% Retained'] = 100 - sieve_summary['% Passing']

    def find_D_value(percent_passing, target_percent):
        return np.interp(target_percent, percent_passing, bin_edges[:-1])

    D10 = find_D_value(sieve_summary['% Passing'], 10)
    D30 = find_D_value(sieve_summary['% Passing'], 30)
    D60 = find_D_value(sieve_summary['% Passing'], 60)
    Cu = D60 / D10
    Cc = (D30 ** 2) / (D10 * D60)

    st.markdown(f"**Coefficient of Uniformity (Cu)**: {Cu:.2f}")
    st.markdown(f"**Coefficient of Curvature (Cc)**: {Cc:.2f}")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(bin_edges[:-1], sieve_summary['% Passing'], marker='o', linestyle='-', color='b', label='% Passing')
    ax1.plot(bin_edges[:-1], sieve_summary['% Retained'], marker='o', linestyle='-', color='r', label='% Retained')

    ax1.set_xlabel('Particle Diameter (mm)')
    ax1.set_ylabel('Cumulative Percentage Passing (%)', color='b')
    ax1.set_title('Gradation Curve')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    ax1.invert_xaxis()
    ax1.set_xscale('log')
    ax1.set_xlim(left=bin_edges.min(), right=bin_edges.max())
    
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.bar(sieve_summary.index.astype(float), sieve_summary['Mass (g)'], width=0.1, alpha=0.6, color='green', label='Mass (g)')
    ax2.set_ylabel('Mass (g)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    mass_distribution = np.repeat(sieve_summary.index.astype(float).values, sieve_summary['Mass (g)'].astype(int))
    density, bins, _ = ax2.hist(mass_distribution, bins=bin_edges, density=True, alpha=0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ax2.plot(bin_centers, density, color='black', linewidth=2, label='Distribution')

    fig.tight_layout()
    st.pyplot(fig)
    
    return sieve_summary

# ---- Streamlit UI ----
st.set_page_config(page_title="Automated Rock Fragment Analysis & Sieve Gradation Tool", layout="wide")
st.title("Automated Rock Fragment Analysis & Sieve Gradation Tool")



# Initialize session state variables
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'contour_data' not in st.session_state:
    st.session_state.contour_data = None
if 'show_extraction' not in st.session_state:
    st.session_state.show_extraction = False
if 'show_gradation' not in st.session_state:
    st.session_state.show_gradation = False
if 'contour_df' not in st.session_state:
    st.session_state.contour_df = None

# Create tabs for the three main steps
tab1, tab2, tab3 = st.tabs(["Step 1: Contour Analysis", "Step 2: Data Extraction", "Step 3: Gradation Curve"])

with tab1:
    st.header("Step 1: Contour Analysis")
    
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    min_area_threshold = st.slider("Minimum Area Threshold (px)", 10, 1000, 100)
    pixels_per_cm = st.number_input("Pixels per cm", min_value=1.0, value=80.0)
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        st.subheader("Original Image")
        st.image(image, channels="BGR", use_column_width=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            analyze_button = st.button("🔍 Step 1: Analyze Image")
        
        if analyze_button:
            processed_img, results = detect_and_measure_contours(image.copy(), min_area_threshold, pixels_per_cm)
            st.session_state.processed_image = processed_img
            
            st.subheader("Processed Image with Contours")
            st.image(processed_img, channels="BGR", use_column_width=True)
            
            if results:
                st.subheader("Contour Analysis Results")
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Save processed image for download
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    cv2.imwrite(temp_file.name, processed_img)
                    with open(temp_file.name, "rb") as file:
                        st.download_button(label="📥 Download Contoured Image",
                                       data=file,
                                       file_name="contoured_image.jpg",
                                       mime="image/jpeg")
                
                # Ask user if they're satisfied with the contours
                st.session_state.show_extraction = True
                st.success("✅ Contour analysis complete! Proceed to Step 2: Data Extraction")
            else:
                st.warning("⚠️ No contours found. Try adjusting the area threshold.")

with tab2:
    st.header("Step 2: Data Extraction")
    st.info("First complete Step 1: Contour Analysis, then proceed here to extract detailed data.")
    
    # Either use the processed image from step 1 or upload a contoured image
    if st.session_state.processed_image is None:
        st.write("Upload a contoured image (with green contours):")
        contoured_image_file = st.file_uploader("Upload contoured image", type=["jpg", "jpeg", "png"])
        if contoured_image_file:
            file_bytes = np.asarray(bytearray(contoured_image_file.read()), dtype=np.uint8)
            st.session_state.processed_image = cv2.imdecode(file_bytes, 1)
            st.image(st.session_state.processed_image, channels="BGR", caption="Uploaded Contoured Image", use_column_width=True)
    else:
        st.image(st.session_state.processed_image, channels="BGR", caption="Image with Detected Contours", use_column_width=True)
    
    density_input = st.text_input("Enter the density of the material (g/cm³):", "2.65")
    pixels_per_mm = st.number_input("Pixels per mm", min_value=0.1, value=8.0)
    
    extract_button = st.button("📊 Step 2: Extract Data")
    
    if extract_button and st.session_state.processed_image is not None:
        try:
            density_g_per_cm3 = float(density_input)
            contour_details = extract_contour_data(st.session_state.processed_image, pixels_per_mm, density_g_per_cm3)
            
            if contour_details:
                st.session_state.contour_df = pd.DataFrame(contour_details)
                st.dataframe(st.session_state.contour_df)
                
                # Save to CSV for download
                csv = st.session_state.contour_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Contour Data (CSV)", 
                                  data=csv, 
                                  file_name="contour_details.csv", 
                                  mime="text/csv")
                
                st.session_state.show_gradation = True
                st.success("✅ Data extraction complete! Proceed to Step 3: Gradation Curve")
            else:
                st.warning("⚠️ No valid contours found in the image.")
        except ValueError:
            st.error("❌ Please enter a valid number for density.")
    elif extract_button and st.session_state.processed_image is None:
        st.error("❌ Please complete Step 1 or upload a contoured image first.")

with tab3:
    st.header("Step 3: Gradation Curve")
    st.info("Complete Steps 1 and 2 first, or upload a CSV file with contour data.")
    
    # Either use the contour data from step 2 or upload a CSV
    if st.session_state.contour_df is None:
        st.write("Upload contour data CSV file:")
        csv_file = st.file_uploader("Upload CSV file", type=["csv"])
        if csv_file:
            st.session_state.contour_df = pd.read_csv(csv_file)
            st.write("Uploaded Data:")
            st.dataframe(st.session_state.contour_df)
    else:
        st.write("Contour Data from Step 2:")
        st.dataframe(st.session_state.contour_df)
    
    gradation_button = st.button("📈 Step 3: Generate Gradation Curve")
    
    if gradation_button and st.session_state.contour_df is not None:
        sieve_summary = generate_gradation_curve(st.session_state.contour_df)
        
        if sieve_summary is not None:
            st.subheader("Sieve Analysis Results")
            st.dataframe(sieve_summary)
            
            # Save to Excel for download
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.contour_df.to_excel(writer, sheet_name='Contour Data', index=False)
                sieve_summary.to_excel(writer, sheet_name='Sieve Analysis', index=True)
            
            st.download_button(label="📥 Download Complete Analysis (Excel)",
                              data=buffer.getvalue(),
                              file_name="rock_fragment_analysis.xlsx",
                              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            st.success("✅ Analysis complete! You can download the full report.")
    elif gradation_button and st.session_state.contour_df is None:
        st.error("❌ Please complete Step 2 or upload contour data CSV first.")

# Add a footer with instructions
st.markdown("---")
st.markdown("""
### Instructions:
1. **Step 1**: Upload an image and analyze it to detect rock fragment contours
2. **Step 2**: Enter the material density to extract detailed measurements from contours
3. **Step 3**: Generate the gradation curve and sieve analysis results
""")