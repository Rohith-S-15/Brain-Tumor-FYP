from fpdf import FPDF
import qrcode
import tempfile
import os
from datetime import datetime

class MedicalReport(FPDF):
    def header(self):
        # Professional Header - Hospital Logo Area and Name
        self.set_fill_color(20, 50, 90) # Darker professional blue
        self.rect(0, 0, 210, 35, 'F')
        
        self.set_y(10)
        self.set_font('Arial', 'B', 18)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, 'NATIONAL INSTITUTE OF NEURO-ONCOLOGY', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 5, 'Department of Advanced Radiology & AI Diagnostics', 0, 1, 'C')
        self.cell(0, 5, '123 Medical Center Blvd | Contact: (555) 019-8372', 0, 1, 'C')
        
        # Reset text color for the body
        self.set_text_color(0, 0, 0)
        self.ln(15)

    def footer(self):
        self.set_y(-20)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.line(10, 277, 200, 277) # Add a line above footer
        self.cell(0, 5, 'CONFIDENTIAL MEDICAL RECORD', 0, 1, 'C')
        self.cell(0, 5, f'Page {self.page_no()} of {{nb}} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'C')

def create_pdf(data):
    pdf = MedicalReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # --- REPORT TITLE ---
    pdf.set_font("Arial", "B", 14)
    # Give some space for the QR code on the right
    pdf.cell(0, 10, 'MRI BRAIN (TUMOR PROTOCOL) - AI VOLUMETRIC ANALYSIS', 0, 1, 'C')
    
    # --- QR CODE ---
    qr = qrcode.make(f"ID:{data['pid']}|Status:{data['alert_title']}")
    qr_path = tempfile.mktemp(suffix=".png")
    qr.save(qr_path)
    # Position QR code slightly lower, on the right
    pdf.image(qr_path, x=175, y=36, w=22)
    
    # Add vertical space so the table doesn't overlap the QR code
    pdf.ln(15)

    # --- PATIENT INFO BLOCK ---
    pdf.set_font("Arial", "B", 11)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 7, " PATIENT DEMOGRAPHICS", border=1, ln=1, fill=True)
    
    pdf.set_font("Arial", "B", 9)
    # Row 1
    y_before = pdf.get_y()
    pdf.cell(30, 6, "Patient Name:", border='L')
    pdf.set_font("Arial", "", 9)
    pdf.cell(50, 6, data['name'])
    
    pdf.set_font("Arial", "B", 9)
    pdf.cell(30, 6, "Patient ID:")
    pdf.set_font("Arial", "", 9)
    pdf.cell(80, 6, data['pid'], border='R', ln=1)
    
    # Row 2
    pdf.set_font("Arial", "B", 9)
    pdf.cell(30, 6, "Age/Sex:", border='L')
    pdf.set_font("Arial", "", 9)
    pdf.cell(50, 6, f"{data['age']} / {data['sex']}")
    
    pdf.set_font("Arial", "B", 9)
    pdf.cell(30, 6, "Exam Date:")
    pdf.set_font("Arial", "", 9)
    pdf.cell(80, 6, data['date_b'], border='R', ln=1)
    
    # Row 3
    pdf.set_font("Arial", "B", 9)
    pdf.cell(30, 6, "Ref. Physician:", border='L')
    pdf.set_font("Arial", "", 9)
    pdf.cell(50, 6, "Dr. A. Neurosurgeon") # Mock physician
    
    pdf.set_font("Arial", "B", 9)
    pdf.cell(30, 6, "Report Date:")
    pdf.set_font("Arial", "", 9)
    pdf.cell(80, 6, datetime.now().strftime('%Y-%m-%d'), border='R', ln=1)

    # Row 4 (Symptoms/Indication)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(30, 6, "Indication:", border='LB')
    pdf.set_font("Arial", "", 9)
    pdf.cell(160, 6, data['symptoms'], border='RB', ln=1)
    
    pdf.ln(5)

    # --- TECHNICAL PARAMETERS ---
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 6, "TECHNIQUE:", ln=1)
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(0, 5, "Multiplanar, multisequence magnetic resonance imaging of the brain was performed before and after the intravenous administration of gadolinium-based contrast material. 3D high-resolution T1, T2, and FLAIR sequences were acquired. Volumetric assessment performed via Deep Learning NeuroForecast AI.")
    pdf.ln(5)

    # --- CLINICAL FINDINGS (Morphology) ---
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 6, "MORPHOLOGY & TISSUE CHARACTERISTICS:", ln=1)
    pdf.set_font("Arial", "", 9)
    sphericity_txt = f"{data['sphericity']:.2f}" if 'sphericity' in data else "N/A"
    necrosis_txt = data['necrosis'] if 'necrosis' in data else "N/A"
    findings_text = (
        f"A mass lesion is identified in the cerebellopontine angle. "
        f"The lesion demonstrates a sphericity index of {sphericity_txt} and exhibits {necrosis_txt.lower()} internal necrosis/cystic changes. "
        f"There is varying degree of contrast enhancement. "
        f"Adjacent brainstem compression and cranial nerve displacement should be clinically correlated."
    )
    pdf.multi_cell(0, 5, findings_text)
    pdf.ln(5)

    # --- VOLUMETRIC ANALYSIS TABLE ---
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 7, " LONGITUDINAL VOLUMETRIC ANALYSIS & FORECAST", border=1, ln=1, fill=True)
    
    # Header Row
    pdf.set_font("Arial", "B", 9)
    pdf.cell(45, 8, "Measurement", border=1, align='C')
    pdf.cell(45, 8, f"Scan 1 ({data['date_a']})", border=1, align='C')
    pdf.cell(50, 8, f"Scan 2 (Current: {data['date_b']})", border=1, align='C')
    pdf.cell(50, 8, f"AI Forecast ({data['date_c']})", border=1, ln=1, align='C')
    
    # Data Row 1 (Volume)
    pdf.set_font("Arial", "", 9)
    pdf.cell(45, 8, "Tumor Volume (cm3)", border=1, align='C')
    pdf.cell(45, 8, f"{data['vol_a']}", border=1, align='C')
    pdf.set_font("Arial", "B", 9)
    pdf.cell(50, 8, f"{data['vol_b']}", border=1, align='C')
    pdf.set_font("Arial", "B", 9)
    if data['growth_bc'] > 10:
        pdf.set_text_color(186, 26, 26)
    elif data['growth_bc'] < -10:
        pdf.set_text_color(0, 128, 0)
    else:
        pdf.set_text_color(0, 101, 145)
    pdf.cell(50, 8, f"{data['vol_c']}", border=1, ln=1, align='C')
    pdf.set_text_color(0, 0, 0)
    
    # Data Row 2 (Metrics)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(45, 8, "Growth/Kinetics", border=1, align='C')
    pdf.set_font("Arial", "", 9)
    pdf.cell(45, 8, "-", border=1, align='C')
    pdf.cell(50, 8, f"Interval Growth: {data['growth_ab']}%", border=1, align='C')
    
    pdf.set_font("Arial", "B", 9)
    if data['growth_bc'] > 10:
        pdf.set_text_color(186, 26, 26)
    elif data['growth_bc'] < -10:
        pdf.set_text_color(0, 128, 0)
    else:
        pdf.set_text_color(0, 101, 145)
    pdf.cell(50, 8, f"Proj. Growth: {data['growth_bc']}%", border=1, ln=1, align='C')
    pdf.set_text_color(0, 0, 0)

    # Extra info
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 6, f"Calculated Tumor Doubling Time: {data['doubling_time']} | AI Forecast Confidence Score: {data['confidence']}%", ln=1)
    pdf.ln(2)

    # --- CLINICAL IMPRESSION (Alerts) ---
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(220, 230, 240)
    pdf.cell(0, 7, " IMPRESSION", border=1, ln=1, fill=True)
    pdf.set_font("Arial", "B", 10)
    
    if data['growth_bc'] > 10:
        pdf.set_text_color(186, 26, 26) # Red
    elif data['growth_bc'] < -10:
        pdf.set_text_color(0, 128, 0) # Green
    else:
        pdf.set_text_color(0, 101, 145) # Blue
        
    pdf.cell(0, 8, f" {data['alert_title']}", border='L R T', ln=1)
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, f" {data['alert_desc']}", border='L R B')
    pdf.ln(5)

    # --- GRAPH ---
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 7, " LONGITUDINAL TRAJECTORY", border=1, ln=1, fill=True)
    # Determine the Y position for the image so it doesn't overlap
    y_graph = pdf.get_y() + 2
    pdf.image(data['imgs']['graph'], x=25, y=y_graph, w=160)
    
    # --- PAGE 2: RADIOGRAPHY ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 7, " RADIOGRAPHIC CROSS-SECTIONS (AXIAL MAX EXTENT)", border=1, ln=1, fill=True)
    pdf.ln(5)
    
    # 3-Column Image Layout
    y_img = pdf.get_y()
    
    # Column 1
    pdf.set_font("Arial", "B", 9)
    pdf.text(25, y_img, f"1. Baseline ({data['date_a']})")
    pdf.image(data['imgs']['img_a'], x=15, y=y_img + 3, w=55)
    
    # Column 2
    pdf.text(85, y_img, f"2. Current ({data['date_b']})")
    pdf.image(data['imgs']['img_b'], x=75, y=y_img + 3, w=55)
    
    # Column 3
    pdf.text(145, y_img, f"3. AI Forecast ({data['date_c']})")
    pdf.image(data['imgs']['img_pred'], x=135, y=y_img + 3, w=55)

    # Add spacing after images
    pdf.set_y(y_img + 65)

    # --- SIGNATURE BLOCK ---
    pdf.ln(10)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(100, 5, "Electronically Generated By:", 0, 0)
    pdf.cell(90, 5, "Reviewing Radiologist/Physician:", 0, 1)
    
    pdf.set_font("Arial", "", 9)
    pdf.cell(100, 5, "NeuroForecast AI System (v2.1)", 0, 0)
    pdf.cell(90, 5, "___________________________________", 0, 1)
    
    pdf.ln(2)
    pdf.cell(100, 5, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 0)
    pdf.cell(90, 5, "MD / DO / MBBS", 0, 1)
    
    pdf.ln(10)
    # Disclaimer
    pdf.set_font("Arial", "I", 7)
    pdf.set_text_color(100, 100, 100)
    disclaimer = "DISCLAIMER: This report is generated by an Artificial Intelligence system for investigational and clinical decision support purposes. It does not constitute a final medical diagnosis. The quantitative volumetric data and morphological assessments are approximations based on algorithmic segmentation. All volumetric forecasts should be corroborated with clinical findings, patient history, and expert radiological review prior to surgical or radiotherapeutic intervention. Not FDA cleared for standalone diagnostic use."
    pdf.multi_cell(0, 3, disclaimer)

    output_path = tempfile.mktemp(suffix=".pdf")
    pdf.output(output_path)
    return output_path