import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask

from flask import Flask, request, render_template, send_file
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

app = Flask(__name__)

# Define the path for saving uploaded files and output files
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs, ax, title=None, show_axis=True):
    colours = ['black']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    if not show_axis:
        ax.axis('off')

def is_circle(contour, approx, circularity_tolerance=0.3, area_ratio_tolerance=0.3):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter ** 2)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    enclosing_circle_area = np.pi * (radius ** 2)
    area_ratio = area / enclosing_circle_area
    is_circular = (1 - circularity_tolerance <= circularity <= 1 + circularity_tolerance)
    is_area_close = (1 - area_ratio_tolerance <= area_ratio <= 1 + area_ratio_tolerance)
    return is_circular and is_area_close

def is_star(approx):
    if len(approx) == 10:
        angles = []
        for i in range(len(approx)):
            pt1 = approx[i][0]
            pt2 = approx[(i + 2) % len(approx)][0]
            angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
            angles.append(angle)
        angle_diff = np.diff(angles)
        if np.all(np.abs(angle_diff) > 0.5):
            return True
    return False

def is_nearly_straight_line(pt1, pt2, pt3, threshold=0.3):
    vec1 = np.array(pt1) - np.array(pt2)
    vec2 = np.array(pt3) - np.array(pt2)
    angle = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))
    return np.abs(angle - np.pi) < threshold

def merge_collinear_points(approx, threshold=0.3):
    new_approx = []
    num_points = len(approx)
    i = 0
    while i < num_points:
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % num_points][0]
        pt3 = approx[(i + 2) % num_points][0]
        if is_nearly_straight_line(pt1, pt2, pt3, threshold):
            new_approx.append(approx[(i) % num_points])
            new_approx.append(approx[(i + 2) % num_points])
            i += 2
        else:
            new_approx.append(approx[i])
            i += 1
    return np.array(new_approx)

def contour_properties(approx):
    x, y, w, h = cv2.boundingRect(approx)
    center = (x + w // 2, y + h // 2)
    aspect_ratio = float(w) / h
    return center, aspect_ratio, w, h

def is_similar(contour1_props, contour2_props):
    center1, aspect_ratio1, w1, h1 = contour1_props
    center2, aspect_ratio2, w2, h2 = contour2_props
    center_dist = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    aspect_ratio_similar = abs(aspect_ratio1 - aspect_ratio2) < 10
    dimension_similar = abs(w1 - w2) < 100 and abs(h1 - h2) < 100
    center_similar = center_dist < 10
    return aspect_ratio_similar and dimension_similar and center_similar

def detect_symmetries(contour, image, tolerance=0.02, angles=np.arange(0, 360, 3)):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    def resize_mask(m, size):
        return cv2.resize(m, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    def check_symmetry(m1, m2):
        if m1.shape != m2.shape:
            m2 = resize_mask(m2, m1.shape)
        return np.mean(np.abs(m1 - m2)) < tolerance * 255
    symmetries = 0
    mask_h, mask_w = mask.shape
    flip_h = cv2.flip(mask, 0)
    flip_v = cv2.flip(mask, 1)
    if check_symmetry(mask, flip_h):
        symmetries += 1
    if check_symmetry(mask, flip_v):
        symmetries += 1
    flip_d1 = cv2.transpose(mask)
    flip_d1 = cv2.flip(flip_d1, 1)
    flip_d2 = cv2.transpose(mask)
    flip_d2 = cv2.flip(flip_d2, 0)
    if check_symmetry(mask, flip_d1):
        symmetries += 1
    if check_symmetry(mask, flip_d2):
        symmetries += 1
    for angle in angles:
        M = cv2.getRotationMatrix2D((mask_w / 2, mask_h / 2), angle, 1)
        rotated_mask = cv2.warpAffine(mask, M, (mask_w, mask_h), flags=cv2.INTER_NEAREST)
        if check_symmetry(mask, rotated_mask):
            symmetries += 1
    return symmetries

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            return render_template('index.html', message="No file part")
        file = request.files['csv_file']
        if file.filename == '':
            return render_template('index.html', message="No selected file")
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            output_data = read_csv(filepath)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            plot(output_data, ax, show_axis=False)
            png_path = os.path.join('uploads', 'Polylines.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            img = cv2.imread(png_path)
            imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thrash = cv2.threshold(imgGry, 240, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            output_img = np.ones_like(img) * 255
            symmetries_img = np.ones_like(img) * 255
            drawn_contours_props = []

            for contour in contours:
                if len(contour) < 5:
                    continue
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                approx = merge_collinear_points(approx)

                contour_props = contour_properties(approx)

                if is_circle(contour, approx):
                    cv2.drawContours(output_img, [approx], 0, (0, 255, 0), 2)
                elif is_star(approx):
                    cv2.drawContours(output_img, [approx], 0, (255, 0, 0), 2)
                else:
                    cv2.drawContours(output_img, [approx], 0, (255, 255, 0), 2)
                
                num_symmetries = detect_symmetries(approx, img)
                cv2.putText(symmetries_img, f'Symmetries: {num_symmetries}', (contour_props[0][0] - 10, contour_props[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
                drawn_contours_props.append(contour_props)

            result_path = os.path.join('uploads', 'DetectedShapes.png')
            cv2.imwrite(result_path, output_img)

            symmetries_path = os.path.join('uploads', 'DetectedSymmetries.png')
            cv2.imwrite(symmetries_path, symmetries_img)

            return render_template('result.html', image_url='/uploads/DetectedShapes.png', symmetries_url='/uploads/DetectedSymmetries.png', message="Processing Complete")

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join('uploads', filename), as_attachment=True)

if __name__ == '__main__':
    # Specify a port, for example 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
