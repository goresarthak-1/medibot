# Installation Guide for PDF Receipt Feature

## Install the Required Package

Run this command to install reportlab for PDF generation:

```bash
pipenv install reportlab
```

## How to Use the PDF Receipt Feature

1. **Login** to your MediBot account
2. **Chat** with the bot about your medical concerns
3. In the **sidebar**, you'll see a new section: "📄 Medical Receipt"
4. Click **"📥 Generate PDF Receipt"** button
5. A **"⬇️ Download Receipt"** button will appear
6. Click to download your medical consultation receipt as PDF

## What's Included in the PDF Receipt

- **Patient Name**: Your username
- **Date & Time**: When the receipt was generated
- **Symptoms/Complaints**: Questions you asked about symptoms
- **Recommendations/Remedies**: AI-provided advice and remedies
- **Medicines/Medications**: Any medication information discussed
- **Disclaimer Warning**: Important notice to consult a certified doctor

## Important Notes

⚠️ The PDF includes a disclaimer stating this is AI-generated and NOT a medical prescription. Always consult a certified doctor before taking any medication.
