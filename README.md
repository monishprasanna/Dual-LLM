# Dual LLM Workbench & Local Patient Model Arena

This repository contains a full dual-model web application, CORS proxy server, and local patient dataset inference engine designed to work with both **LM Studio** and **Local Python PyTorch / Dataset Engines**.

---

## 🌟 Features Implemented

1. **Dual Model Workbench & Inter-LLM Auto Conversation Web App** (`lm_studio_arena.html`)
   - **Individual Prompt Mode**: Send direct prompts to Model A and Model B in side-by-side chat windows.
   - **Inter-LLM Auto Conversation Mode**: Make Model A (e.g. Local Patient Model) and Model B (e.g. LM Studio uncensored model) converse back-and-forth automatically for N turns.
   - **Multi-API Endpoint Format Support**:
     - `🐍 Local Patient Model (Direct Python /api/local_model/chat)`
     - `LM Studio Native (/api/v1/chat)`
     - `OpenAI Compatible (/v1/chat/completions)`
   - **Export Transcript**: Export complete inter-model dialogue to Markdown `.md`.

2. **Backend Proxy & Local Model Server** (`server.py` & `local_patient_model.py`)
   - Eliminates browser CORS errors automatically.
   - Runs direct Python inference on local patient records (`patients.csv` / `patient_finetune_dataset.jsonl`) or local PyTorch LoRA adapters (`lora_adapter`).

---

## 🚀 How to Start / Restart the Application

### 1. Launch the Server
Open PowerShell or Terminal in this directory and run:

```powershell
python server.py
```

*Output:*
```text
Proxy Server running at http://localhost:8000
```

---

### 2. Open the Web Application
Open your web browser and navigate to:

👉 **[http://localhost:8000/lm_studio_arena.html](http://localhost:8000/lm_studio_arena.html)**  
*(or simply [http://localhost:8000](http://localhost:8000))*

---

### 3. Connect LM Studio (Optional for Model B)
If you want Model B (or Model A) to talk to LM Studio:
1. Open LM Studio on your machine.
2. Load your model (e.g. `llama-3.2-3b-instruct-uncensored`).
3. Start the **LM Studio Local Server** on port `1234` (`http://localhost:1234/v1`).
4. Click **"🔄 Fetch LM Studio Models"** in the web app settings to auto-detect loaded models!

---

## ⏸️ How to Pause / Stop the Server

To stop the background server at any time:
- In your terminal window, press **`Ctrl + C`**.

---

## 📁 Key File Structure

| File | Description |
| :--- | :--- |
| **`server.py`** | Local Python HTTP server (Port 8000) providing CORS proxy and `/api/local_model/chat` API. |
| **`local_patient_model.py`** | PyTorch/PEFT & fallback dataset inference engine for patient records. |
| **`lm_studio_arena.html`** | Web application frontend (UI, Individual Chat, Auto Arena, Markdown export). |
| **`patients.csv`** | Synthetic patient records dataset (P-1001 to P-1019). |
| **`patient_finetune_dataset.jsonl`** | Formatted JSONL dataset for LoRA fine-tuning. |
| **`train.py`** | Unsloth + LoRA fine-tuning script for training Llama-3.2 on local GPU. |
| **`auto_chat.py`** | Command-line CLI script for automated back-and-forth dialogue. |
