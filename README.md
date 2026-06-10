# Sentin-Edge AI

**Fault-Tolerant Multi-Modal Cascaded Psychiatric Screening System**
*Fully offline, localhost 127.0.0.1, zero cloud dependency, HIPAA-safe*

## Architecture Overview

3-Tier Progressive Gating Pipeline:
- **Tier 1:** LightGBM behavioral risk classifier (DASS-42 tabular data)
- **Tier 1.5:** Librosa acoustic biomarker validation (triggered only on High Risk)
- **Tier 2:** EEG brainwave biometric authority (chunked parallel processing)

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Dummy DASS-42 Dataset:**
   ```bash
   python tier1_data_prep.py
   ```

3. **Train Tier 1 Model (LightGBM):**
   ```bash
   python tier1_classifier.py
   ```

4. **Run the Streamlit Application:**
   ```bash
   streamlit run app.py --server.address 127.0.0.1 --server.port 8501
   ```

## Security & Compliance
- All inference runs on localhost 127.0.0.1 exclusively.
- No patient data written to disk.
- Audio captured and analyzed in-memory.
- Streamlit configured in headless mode with CORS disabled.
