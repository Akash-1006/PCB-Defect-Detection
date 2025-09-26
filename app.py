from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import uuid, shutil, glob, os
import json, csv
from fpdf import FPDF

app = Flask(__name__)

# --------- CONFIG ----------
UPLOAD_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model once at startup
yolo_model = YOLO("best.pt")
# ---------------------------

# ---------------------------
# Utility Functions
# ---------------------------

def compute_severity(bbox, defect_class, critical_mask=None, num_defects=1):
    class_severity = {
        'Missing_hole': 9,
        'Short': 9,
        'Open_circuit': 8,
        'Spurious_copper': 5,
        'Mouse_bite': 4,
        'Spur': 4
    }
    base = class_severity.get(defect_class, 3)  # now matches YOLO classes exactly
    x, y, w, h = bbox
    area = w * h
    area_modifier = min(area/(500*500),1)*2
    overlap_modifier = 0
    if critical_mask is not None:
        roi = critical_mask[y:y+h, x:x+w]
        overlap_modifier = np.sum(roi>0)/(w*h)*2
    severity = base + area_modifier + overlap_modifier
    return round(min(severity,10))


def predict_root_cause(defect_class):
    mapping = {
        'Missing_hole': 'Drilling issue',
        'Spurious_copper': 'Etching problem',
        'Short': 'Soldering/printing issue',
        'Open_circuit': 'Broken trace / PCB manufacturing defect',
        'Mouse_bite': 'Etching / copper removal issue',
        'Spur': 'Etching / unwanted copper',
    }
    return mapping.get(defect_class, 'Unknown')


def generate_report(pcb_id, defects_info, output_dir):
    json_path = os.path.join(output_dir, f"{pcb_id}_report.json")
    with open(json_path, 'w') as f:
        json.dump({"PCB_ID": pcb_id, "defects": defects_info}, f, indent=4)

    csv_path = os.path.join(output_dir, f"{pcb_id}_report.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["class", "bbox", "confidence", "severity", "root_cause"])
        writer.writeheader()
        for d in defects_info:
            writer.writerow(d)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"PCB ID: {pcb_id}", ln=True)
    for defect in defects_info:
        pdf.cell(0, 10, f"{defect['class']} - Severity: {defect['severity']} - Cause: {defect['root_cause']}", ln=True)
    pdf_path = os.path.join(output_dir, f"{pcb_id}_report.pdf")
    pdf.output(pdf_path)

    return {"json": json_path, "csv": csv_path, "pdf": pdf_path}

def batch_statistics(batch_defects):
    total_boards = len(batch_defects)
    defect_free = sum(1 for d in batch_defects if len(d['defects']) == 0)
    all_defect_types = [d['class'] for board in batch_defects for d in board['defects']]
    most_common_defect = max(set(all_defect_types), key=all_defect_types.count) if all_defect_types else None
    return {
        "total_boards": total_boards,
        "defect_free_percentage": round(defect_free/total_boards*100,2),
        "most_common_defect": most_common_defect
    }

# ---------------------------
# Flask Routes
# ---------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    method = request.form.get('method', '')

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    # ===== YOLO Detection =====
    if method == 'yolo':
        return analyze_single_image(input_path, filename)

    # ===== Rule-based AOI =====
    elif method == 'rule':
        processed_path = run_rule_based_aoi(input_path, app.config['UPLOAD_FOLDER'])
        rel_path = os.path.relpath(processed_path,
                                   start=os.path.join(app.root_path, 'static')).replace("\\", "/")
        processed_url = url_for('static', filename=rel_path)
        return jsonify({'processed_image_url': processed_url})

    else:
        return jsonify({'error': 'Invalid method'}), 400

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400

    files = request.files.getlist('images')
    method = request.form.get('method', 'yolo')

    batch_results = []
    for file in files:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        if method == 'yolo':
            single_result = analyze_single_image(input_path, filename)
            batch_results.append(single_result)
        else:
            # For rule-based, just process image
            processed_path = run_rule_based_aoi(input_path, app.config['UPLOAD_FOLDER'])
            rel_path = os.path.relpath(processed_path,
                                       start=os.path.join(app.root_path, 'static')).replace("\\", "/")
            processed_url = url_for('static', filename=rel_path)
            batch_results.append({"processed_image_url": processed_url, "defects": []})

    # Compute batch stats
    stats = batch_statistics(batch_results)

    return jsonify({
        "batch_results": batch_results,
        "batch_stats": stats
    })

# ---------------------------
# Helper: Analyze Single Image
# ---------------------------

def analyze_single_image(input_path, filename):
    yolo_results = yolo_model.predict(
        source=input_path,
        conf=0.25,
        save=True,
        project=app.config['UPLOAD_FOLDER'],
        name='yolo',
        exist_ok=True
    )

    output_dir = Path(yolo_results[0].save_dir)
    files = glob.glob(str(output_dir / "*.jpg"))
    if not files:
        return {"error": "No YOLO output image found", "defects": []}
    default_output = max(files, key=os.path.getmtime)
    final_output = output_dir / f"processed_{uuid.uuid4().hex}.jpg"
    shutil.move(default_output, final_output)

    # --- Severity + Root Cause ---
    defects_info = []
    for result in yolo_results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            bbox = [int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])]
            defect_class = yolo_model.names[int(cls)]
            severity = compute_severity(bbox, defect_class, critical_mask=None, num_defects=len(result.boxes))
            root_cause = predict_root_cause(defect_class)
            defects_info.append({
                "class": defect_class,
                "bbox": bbox,
                "confidence": float(conf),
                "severity": severity,
                "root_cause": root_cause
            })

    # --- Generate Reports ---
    report_paths = generate_report(filename.split('.')[0], defects_info, app.config['UPLOAD_FOLDER'])

    rel_path = os.path.relpath(final_output,
                               start=os.path.join(app.root_path, 'static')).replace("\\", "/")
    processed_url = url_for('static', filename=rel_path)

    return {"processed_image_url": processed_url, "defects": defects_info, "report_paths": report_paths}

# ---------------------------
# Rule-based AOI (existing)
# ---------------------------
def run_rule_based_aoi(test_path: str, output_dir: str) -> str:
    TEMPLATE_PATH = r"C:\Users\Akash Balaji\Downloads\template_pcb.jpg"
    template_img = cv2.imread(TEMPLATE_PATH)
    test_img = cv2.imread(test_path)
    if template_img is None or test_img is None:
        raise FileNotFoundError("Template or uploaded image not found!")

    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    med_template = cv2.medianBlur(gray_template, 7)
    med_test = cv2.medianBlur(gray_test, 7)
    gaus_template = cv2.GaussianBlur(med_template, (3,3),1)
    gaus_test = cv2.GaussianBlur(med_test, (3,3),1)
    sold_template = cv2.inRange(gaus_template, 140, 255)
    sold_test = cv2.inRange(gaus_test, 145, 255)
    wire_template = cv2.inRange(gaus_template, 95, 145)
    wire_test = cv2.inRange(gaus_test, 105, 150)
    kernel = np.ones((7,7))
    open_wire_template = cv2.morphologyEx(wire_template, cv2.MORPH_OPEN, kernel)
    open_wire_test = cv2.morphologyEx(wire_test, cv2.MORPH_OPEN, kernel)
    kernel_close = np.ones((13,3))
    close_sold_template = cv2.morphologyEx(sold_template, cv2.MORPH_CLOSE, kernel_close)
    close_sold_test = cv2.morphologyEx(sold_test, cv2.MORPH_CLOSE, kernel_close)

    def fill_holes(binary_img):
        fill = binary_img.copy()
        h,w = fill.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(fill, mask, (0,0), 255)
        return cv2.bitwise_not(fill)

    hole_template = fill_holes(close_sold_template)
    hole_test = fill_holes(close_sold_test)
    pd_sold = cv2.morphologyEx(sold_test - sold_template, cv2.MORPH_OPEN, np.ones((3,3)))
    nd_sold = cv2.morphologyEx(sold_template - sold_test, cv2.MORPH_OPEN, np.ones((3,3)))
    pd_wire = cv2.morphologyEx(open_wire_test - open_wire_template, cv2.MORPH_OPEN, np.ones((3,3)))
    nd_wire = cv2.morphologyEx(open_wire_template - open_wire_test, cv2.MORPH_OPEN, np.ones((3,3)))
    pd_hole = hole_test - hole_template
    nd_hole = cv2.morphologyEx(hole_template - hole_test, cv2.MORPH_OPEN, np.ones((3,3)))

    defects = test_img.copy()
    defects[pd_sold==255] = [0,0,255]
    defects[pd_wire==255] = [0,0,255]
    defects[pd_hole==255] = [0,0,255]
    defects[nd_sold==255] = [255,0,0]
    defects[nd_wire==255] = [255,0,0]
    defects[nd_hole==255] = [255,0,0]

    out_name = 'rule_processed_' + os.path.basename(test_path)
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, defects)
    return out_path

# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
