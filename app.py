import os
import glob
import json
from flask import Flask, render_template, request, redirect, url_for, session, send_file
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import find_objects, label
from scipy.interpolate import make_interp_spline # Added for smooth curves!
from torch.amp import autocast
import matplotlib
matplotlib.use('Agg')

from utils.model_arch import TempVSNet
from utils.pdf_generator import create_pdf

app = Flask(__name__)
app.secret_key = "super_secret_medical_key"

# Paths
LOCAL_DATA_DIR = "local_data/mris"
DEMO_CSV = "local_data/VS-MC-RC2_demographic_Dec2025.csv"
MODELS_DIR = "models"
STATIC_DIR = "static"
CACHE_FILE = "local_data/patients_db.json"
os.makedirs(STATIC_DIR, exist_ok=True)

INDIAN_NAMES_MALE = [
    "Aarav Patel", "Vihaan Sharma", "Vivaan Kumar", "Advik Reddy", "Rohan Mehta",
    "Ishaan Verma", "Arjun Nair", "Karan Iyer", "Rahul Banerjee", "Aditya Bhat"
]
INDIAN_NAMES_FEMALE = [
    "Ananya Singh", "Diya Gupta", "Saanvi Desai", "Kavya Joshi", "Neha Chatterjee",
    "Riya Rao", "Priya Menon", "Pooja Malhotra", "Sneha Kapoor"
]
# Fallback list (used when sex is unknown)
INDIAN_NAMES = INDIAN_NAMES_MALE + INDIAN_NAMES_FEMALE

VS_TREATMENT_OPTIONS = ["Watchful Waiting", "Gamma Knife SRS", "Microsurgery (Retrosigmoid)", "Fractionated Stereotactic Radiotherapy"]
VS_AUDIOMETRY = ["Mild sensorineural hearing loss (SNHL) in the affected ear", "Moderate-to-severe SNHL with speech discrimination score of 60%",
    "Profound SNHL; serviceable hearing in contralateral ear", "Pure tone average (PTA) of 45 dB; word recognition score 72%"]
VS_INITIAL_FINDINGS = [
    "MRI with gadolinium revealed a well-defined intracanalicular mass at the internal auditory canal (IAC), consistent with VS.",
    "Contrast-enhanced MRI demonstrated a small cerebellopontine angle (CPA) mass, measuring approximately 1.2 cm, consistent with VS.",
    "Gadolinium-enhanced MRI confirmed a VS with predominantly intracanalicular component without brainstem compression.",
    "MRI findings revealed a Grade II VS per Koos classification with extension into the CPA cistern."
]
VS_MANAGEMENT_NOTES = [
    "Patient counselled regarding the natural history of VS. Given the small size and mild symptom burden, conservative management with serial MRI surveillance was agreed upon.",
    "Multi-disciplinary team (MDT) review completed. Patient opted for Gamma Knife SRS following detailed risk-benefit discussion. Procedure planned for next quarter.",
    "Given tumor stability over prior interval, continued surveillance strategy maintained. Annual MRI protocol confirmed.",
    "Slight volumetric increase noted on interval imaging. MDT review recommended repeat audiological assessment and re-evaluation in 6 months."
]

def get_patient_details(pid):
    """Returns consistent patient details for a given PID.
    Name is always derived from this function — never from the cache — so it
    stays the same across all pages (database, profile, analysis, mri_viewer).
    Name is gender-appropriate: male names for M patients, female for F.
    """
    age, sex, symptoms = "Unknown", "Unknown", "Not Recorded"
    if os.path.exists(DEMO_CSV):
        try:
            df = pd.read_csv(DEMO_CSV)
            id_col = next((c for c in df.columns if 'id' in c.lower() or 'patient' in c.lower()), None)
            if id_col:
                base_id = "_".join(pid.split("_")[:4])
                row = df[df[id_col].astype(str).str.contains(base_id, na=False)]
                if not row.empty:
                    row = row.iloc[0]
                    age_col = next((c for c in df.columns if 'age' in c.lower()), None)
                    sex_col = next((c for c in df.columns if 'sex' in c.lower() or 'gender' in c.lower()), None)
                    sym_col = next((c for c in df.columns if 'symptom' in c.lower()), None)
                    if age_col: age = str(row[age_col])
                    if sex_col: sex = str(row[sex_col])
                    if sym_col: symptoms = str(row[sym_col])
        except Exception:
            pass

    # Pick a gender-appropriate name using pid hash as a stable seed
    seed = abs(hash(pid))
    if sex == 'M':
        name = INDIAN_NAMES_MALE[seed % len(INDIAN_NAMES_MALE)]
    elif sex == 'F':
        name = INDIAN_NAMES_FEMALE[seed % len(INDIAN_NAMES_FEMALE)]
    else:
        name = INDIAN_NAMES[seed % len(INDIAN_NAMES)]

    return {"name": name, "age": age, "sex": sex, "symptoms": symptoms}

def get_vs_clinical_history(pid, scans):
    """Generate a realistic dummy VS clinical history for a patient based on their scan data."""
    seed = hash(pid)
    rng = lambda lst: lst[abs(seed) % len(lst)]
    history = []
    first_scan = scans[0] if scans else None
    year = first_scan['date_a'][:4] if first_scan else "1992"
    history.append({
        "date": f"{year}-01",
        "type": "Initial Presentation",
        "note": f"Patient presented with progressive unilateral hearing loss and intermittent tinnitus. {rng(VS_INITIAL_FINDINGS)}"
    })
    history.append({
        "date": f"{year}-03",
        "type": "Audiometry Report",
        "note": rng(VS_AUDIOMETRY) + ". Referred to neurotology for further evaluation."
    })
    history.append({
        "date": f"{year}-04",
        "type": "MDT Management Decision",
        "note": f"Recommended treatment: {rng(VS_TREATMENT_OPTIONS)}. " + rng(VS_MANAGEMENT_NOTES)
    })
    for i, scan in enumerate(scans):
        g = scan.get('growth_bc', 0)
        trend = "volumetric stability" if abs(g) <= 10 else ("tumor regression" if g < -10 else "progressive growth")
        history.append({
            "date": scan['date_b'],
            "type": f"Follow-up MRI Review (Session {i+1})",
            "note": f"Interval MRI analysis demonstrated {trend} (growth rate: {g:+.1f}%). Confidence score: {scan.get('confidence', 0):.1f}%. {'MDT review advised.' if abs(g) > 10 else 'Routine surveillance continued.'}"
        })
    history.append({
        "date": "Ongoing",
        "type": "Current Management Plan",
        "note": "Annual gadolinium-enhanced MRI surveillance. Audiological assessments every 12 months. Patient to report any sudden changes in balance, facial weakness, or headache immediately to the neurology department."
    })
    return history

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache_data):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models =[]
print("Loading AI Models into memory...")
for i in range(5):
    path = os.path.join(MODELS_DIR, f"fold{i}_best.pth")
    if os.path.exists(path):
        m = TempVSNet(base_ch=24).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True)['model_state'])
        m.eval()
        models.append(m)
print(f"Loaded {len(models)} models.")

def postprocess(pred):
    if pred.sum() == 0: return pred
    lab, n = label(pred)
    if n == 0: return pred
    sizes =[np.sum(lab == i) for i in range(1, n+1)]
    return (lab == np.argmax(sizes)+1).astype(np.uint8)

@app.route('/')
def index(): return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    role, password = request.form.get('role'), request.form.get('password')
    if password == "admin":
        session['role'] = role
        return redirect(url_for('database'))
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/database')
def database():
    if 'role' not in session: return redirect(url_for('index'))
    npz_files = glob.glob(f"{LOCAL_DATA_DIR}/*.npz")
    cache = load_cache()
    
    # Group by patient ID to show one row per unique patient
    patient_map = {}
    
    for path in sorted(npz_files):
        filename = os.path.basename(path)
        parts = filename.replace(".npz", "").split("_")
        pid = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
        
        status, risk_level = "Pending", "unknown"
        confidence = 0
        growth_bc = 0
        if filename in cache:
            status = "Analyzed"
            g = cache[filename].get('growth_bc', 0)
            growth_bc = g
            confidence = cache[filename].get('confidence', 0)
            if g > 10: 
                risk_level = "High Risk"
            elif g < -10: 
                risk_level = "Regression"
            else: 
                risk_level = "Stable"
        
        demo = get_patient_details(pid)
        if pid not in patient_map:
            patient_map[pid] = {
                "id": pid,
                "name": demo["name"],
                "age": demo["age"],
                "sex": demo["sex"],
                "symptoms": demo["symptoms"],
                "last_date": parts[4],
                "filename": filename,
                "status": status,
                "risk_level": risk_level,
                "confidence": confidence,
                "scan_count": 1
            }
        else:
            patient_map[pid]["scan_count"] += 1
            # Update to latest status, but PRIORITIZE High Risk so it's not hidden
            if status == "Analyzed":
                current_risk = patient_map[pid]["risk_level"]
                # Keep High Risk if it ever occurred, otherwise use the most recent
                if current_risk != "High Risk":
                    patient_map[pid]["status"] = status
                    patient_map[pid]["risk_level"] = risk_level
                    patient_map[pid]["confidence"] = confidence
                
                # Always update filename/date to the latest one, unless it's hiding a high risk scan
                if current_risk != "High Risk" or risk_level == "High Risk":
                    patient_map[pid]["filename"] = filename
                    patient_map[pid]["last_date"] = parts[4]

    patients = list(patient_map.values())
    
    # Calculate stats based on unique patients now
    stats = {"total": len(patients), "analyzed": 0, "high_risk": 0, "stable": 0, "regression": 0}
    for p in patients:
        if p["status"] == "Analyzed":
            stats["analyzed"] += 1
            if p["risk_level"] == "High Risk": stats["high_risk"] += 1
            elif p["risk_level"] == "Regression": stats["regression"] += 1
            elif p["risk_level"] == "Stable": stats["stable"] += 1

    return render_template('database.html', patients=patients, stats=stats, role=session.get('role'))

@app.route('/patient/<pid>')
def patient_profile(pid):
    if 'role' not in session: return redirect(url_for('index'))
    cache = load_cache()
    demo = get_patient_details(pid)
    
    # Collect all scans for this patient
    scans = []
    for filename, data in cache.items():
        if data.get('pid') == pid:
            scans.append(data)
    scans.sort(key=lambda x: x.get('date_a', ''))
    
    # Determine overall risk
    risk_level = "Pending"
    latest_alert = "info"
    if scans:
        g = scans[-1].get('growth_bc', 0)
        if g > 10: risk_level = "High Risk"; latest_alert = "error"
        elif g < -10: risk_level = "Regression"; latest_alert = "success"
        else: risk_level = "Stable"; latest_alert = "info"
    
    clinical_history = get_vs_clinical_history(pid, scans)
    
    # Build scan timeline entries
    scan_timeline = []
    npz_files = glob.glob(f"{LOCAL_DATA_DIR}/*.npz")
    for path in sorted(npz_files):
        filename = os.path.basename(path)
        parts = filename.replace(".npz", "").split("_")
        fpid = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
        if fpid != pid:
            continue
        entry = {
            "filename": filename,
            "date_a": parts[4],
            "date_b": parts[5],
            "date_c": parts[6],
            "status": "Pending",
            "risk_level": "unknown",
            "growth_bc": None
        }
        if filename in cache:
            g = cache[filename].get('growth_bc', 0)
            entry["status"] = "Analyzed"
            entry["growth_bc"] = g
            if g > 10: entry["risk_level"] = "High Risk"
            elif g < -10: entry["risk_level"] = "Regression"
            else: entry["risk_level"] = "Stable"
        scan_timeline.append(entry)
    
    return render_template('patient_profile.html',
        pid=pid, name=demo['name'], age=demo['age'], sex=demo['sex'],
        symptoms=demo['symptoms'], scans=scans, scan_timeline=scan_timeline,
        clinical_history=clinical_history, risk_level=risk_level,
        latest_alert=latest_alert, role=session.get('role')
    )

@app.route('/mri_viewer/<filename>')
def mri_viewer(filename):
    if 'role' not in session: return redirect(url_for('index'))
    cache = load_cache()
    parts = filename.replace(".npz", "").split("_")
    pid = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
    data = cache.get(filename, {})
    imgs = data.get('imgs', {})
    # Always use get_patient_details for consistent name
    demo = get_patient_details(pid)
    return render_template('mri_viewer.html',
        pid=pid, filename=filename,
        date_a=parts[4], date_b=parts[5], date_c=parts[6],
        img_a=imgs.get('img_a',''), img_b=imgs.get('img_b',''), img_pred=imgs.get('img_pred',''),
        name=demo['name'], role=session.get('role')
    )

@app.route('/add_patient', methods=['GET', 'POST'])
def add_patient():
    if 'role' not in session or session['role'] != 'radiologist':
        return "Access Denied.", 403
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.npz'):
            file.save(os.path.join(LOCAL_DATA_DIR, file.filename))
            return redirect(url_for('database'))
    return render_template('add_patient.html', role=session.get('role'))

@app.route('/analysis/<filename>')
def analysis(filename):
    if 'role' not in session: return redirect(url_for('index'))
    
    cache = load_cache()
    parts = filename.replace(".npz", "").split("_")
    pid = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
    date_a, date_b, date_c = parts[4], parts[5], parts[6]
    
    d_a = datetime.strptime(date_a, "%Y-%m-%d")
    d_b = datetime.strptime(date_b, "%Y-%m-%d")
    d_c = datetime.strptime(date_c, "%Y-%m-%d")
    demographics = get_patient_details(pid)

    if filename in cache:
        # Always override the cached name/demographics with the authoritative
        # get_patient_details() result so name is consistent across all pages.
        report_data = dict(cache[filename])
        report_data['name']     = demographics['name']
        report_data['age']      = demographics['age']
        report_data['sex']      = demographics['sex']
        report_data['symptoms'] = demographics['symptoms']
        session['current_report'] = report_data
        return render_template('analysis.html', **report_data, role=session.get('role'), filename=filename)

    filepath = os.path.join(LOCAL_DATA_DIR, filename)
    d = np.load(filepath)
    
    img_a, img_b, diff = d['img_a'].astype(np.float32), d['img_b'].astype(np.float32), d['diff_ab'].astype(np.float32)
    mask_a = (d['mask_a'] > 0.5).astype(np.uint8)
    mask_b = (d['mask_b'] > 0.5).astype(np.uint8)

    t_img_a = torch.from_numpy(img_a).unsqueeze(0).unsqueeze(0).to(device)
    t_img_b = torch.from_numpy(img_b).unsqueeze(0).unsqueeze(0).to(device)
    t_diff  = torch.from_numpy(diff).unsqueeze(0).unsqueeze(0).to(device)
    t_ma    = torch.from_numpy(mask_a).float().unsqueeze(0).unsqueeze(0).to(device)
    t_mb    = torch.from_numpy(mask_b).float().unsqueeze(0).unsqueeze(0).to(device)
    dab = torch.tensor([(d_b - d_a).days], dtype=torch.float32).to(device)
    dbc = torch.tensor([(d_c - d_b).days], dtype=torch.float32).to(device)

    probs, individual_vols = [], []
    vv = float(d['spacing'][0]) * float(d['spacing'][1]) * float(d['spacing'][2]) / 1000.0

    with torch.no_grad():
        for m in models:
            with autocast('cuda'):
                p = torch.sigmoid(m(t_img_a, t_ma, t_img_b, t_mb, t_diff, dab, dbc)).squeeze().cpu().numpy()
                probs.append(p)
                bin_p = postprocess((p > 0.5).astype(np.uint8))
                individual_vols.append(float(np.sum(bin_p)) * vv)
                
    pred_bin = postprocess((np.mean(probs, axis=0) > 0.5).astype(np.uint8))

    vol_a = float(round(float(np.sum(mask_a)) * vv, 3))
    vol_b = float(round(float(np.sum(mask_b)) * vv, 3))
    vol_c = float(round(float(np.sum(pred_bin)) * vv, 3))
    
    growth_ab = float(round(((vol_b - vol_a) / vol_a) * 100, 1)) if vol_a > 0 else 0.0
    growth_bc = float(round(((vol_c - vol_b) / vol_b) * 100, 1)) if vol_b > 0 else 0.0

    # FIX: Bulletproof Confidence Score
    if vol_c > 0 and len(individual_vols) > 1:
        vol_std = float(np.std(individual_vols))
        cv = vol_std / vol_c
        confidence = float(round(max(0.0, min(100.0, (1.0 - cv) * 100.0)), 1))
    else:
        vol_std = 0.0
        confidence = 0.0 if vol_c == 0 else 100.0

    # Tumor Doubling Time
    doubling_time = "N/A (Shrinking/Stable)"
    if vol_c > vol_b and vol_b > 0 and growth_bc > 2.0:
        days = (d_c - d_b).days
        tdt_days = days * (np.log(2) / np.log(vol_c / vol_b))
        doubling_time = f"{round(tdt_days / 30.44, 1)} Months"

    # Diagnosis
    if growth_bc > 10:
        alert_type, alert_title = "error", "PROGRESSIVE GROWTH DETECTED"
        alert_desc = f"AI indicates volumetric expansion >10%. Estimated doubling time: {doubling_time}. Surgical/SRS consultation advised."
    elif growth_bc < -10:
        alert_type, alert_title = "success", "TUMOR REGRESSION DETECTED"
        alert_desc = "AI indicates significant volumetric decrease. Recommend continued surveillance."
    else:
        alert_type, alert_title = "info", "STABLE TUMOR TRAJECTORY"
        alert_desc = "AI indicates stability within standard +/- 10% thresholds. Recommend routine 12-month MRI follow-up."

    # Generate Image Slices
    combined = mask_b | pred_bin
    z = int(np.argmax(combined.sum(axis=(1,2)))) if combined.sum() > 0 else 64
    if combined.sum() > 0:
        slices = find_objects(combined.astype(int))[0]
        ymin, ymax = max(0, slices[1].start-35), min(128, slices[1].stop+35)
        xmin, xmax = max(0, slices[2].start-35), min(128, slices[2].stop+35)
    else:
        ymin, ymax, xmin, xmax = 20, 108, 20, 108

    def save_slice(img, mask, color, out_name):
        fig, ax = plt.subplots(figsize=(4,4), facecolor='black')
        ax.imshow(img[ymin:ymax, xmin:xmax], cmap='gray', vmin=np.percentile(d['img_c'],1), vmax=np.percentile(d['img_c'],99))
        if mask.sum() > 0: ax.contour(mask[ymin:ymax, xmin:xmax], colors=color, linewidths=2)
        ax.axis('off')
        out_path = os.path.join(STATIC_DIR, out_name)
        plt.savefig(out_path, bbox_inches='tight', facecolor='black', pad_inches=0)
        plt.close(fig)

    save_slice(img_a[z], mask_a[z], 'cyan', f"{pid}_scan_a.png")
    save_slice(img_b[z], mask_b[z], 'cyan', f"{pid}_scan_b.png")
    save_slice(d['img_c'][z], pred_bin[z], '#ff3333', f"{pid}_pred.png")

    # --- ADVANCED SMOOTH GRAPH GENERATION ---
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='white')
    dates =[d_a, d_b, d_c]
    vols = [vol_a, vol_b, vol_c]
    date_nums = mdates.date2num(dates)
    
    # Create smooth spline curve
    try:
        x_smooth = np.linspace(date_nums.min(), date_nums.max(), 300)
        spl = make_interp_spline(date_nums, vols, k=2) # k=2 for 3 points
        y_smooth = spl(x_smooth)
        
        # Plot filled area under smooth curve
        ax.fill_between(x_smooth, 0, y_smooth, color='#0ea5e9', alpha=0.1)
        ax.plot(x_smooth, y_smooth, color='#0ea5e9', linewidth=2.5, label='Biological Trajectory')
    except:
        # Fallback if spline fails
        ax.plot(dates, vols, color='#0ea5e9', linewidth=2.5, label='Trajectory')

    # Plot original data points
    ax.plot(dates[:2], vols[:2], 'o', color='#006591', markersize=8, label='Historical Scans')
    ax.plot(dates[2:], vols[2:], 'D', color='#ba1a1a', markersize=8, label='AI Forecast Target')

    # Stability Zone
    ax.axhspan(vol_b * 0.9, vol_b * 1.1, color='gray', alpha=0.15, label='±10% Clinical Stability')
    
    ax.set_ylabel("Volume (cm³)", fontweight='bold', color='#333333')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig.autofmt_xdate()
    plt.tight_layout()
    graph_path = os.path.join(STATIC_DIR, f"{pid}_graph.png")
    plt.savefig(graph_path, dpi=200)
    plt.close(fig)

    # Simulated Morphology Metrics (for project depth)
    sphericity = round(0.85 - (abs(growth_bc)/100), 2)
    sphericity = max(0.4, min(1.0, sphericity))

    report_data = {
        'pid': str(pid), 'name': demographics['name'], 'age': demographics['age'], 
        'sex': demographics['sex'], 'symptoms': demographics['symptoms'],
        'date_a': date_a, 'date_b': date_b, 'date_c': date_c,
        'vol_a': vol_a, 'vol_b': vol_b, 'vol_c': vol_c, 
        'growth_ab': growth_ab, 'growth_bc': growth_bc,
        'doubling_time': doubling_time, 'confidence': confidence,
        'sphericity': sphericity, 'necrosis': 'Low' if growth_bc < 5 else 'Moderate',
        'alert_type': alert_type, 'alert_title': str(alert_title), 'alert_desc': str(alert_desc),
        'imgs': {'img_a': f"static/{pid}_scan_a.png", 'img_b': f"static/{pid}_scan_b.png", 'img_pred': f"static/{pid}_pred.png", 'graph': graph_path}
    }

    cache[filename] = report_data
    save_cache(cache)
    session['current_report'] = report_data

    return render_template('analysis.html', **report_data, role=session.get('role'), filename=filename)

@app.route('/download_pdf')
def download_pdf():
    data = session.get('current_report')
    pdf_path = create_pdf(data)
    return send_file(pdf_path, as_attachment=True, download_name=f"{data['pid']}_Clinical_Report.pdf")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)