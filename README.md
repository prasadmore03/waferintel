# WaferIntel - Intelligent Wafer Defect Detection

## 🎯 Project Overview
WaferIntel is an AI-powered semiconductor manufacturing defect detection system that provides intelligent wafer analysis with real-time defect classification and expert insights.

## 🚀 Features
- **🔬 Defect Detection**: Upload wafer images for real-time AI analysis
- **📊 Model Insights**: Comprehensive model performance metrics and comparisons
- **🤖 AI Assistant**: Expert semiconductor manufacturing guidance
- **📈 Performance Analytics**: Real-time monitoring and statistics
- **🎨 Professional UI**: Modern, responsive interface with gradient styling

## 📋 System Requirements
- Python 3.8+
- TensorFlow 2.x
- Streamlit
- NumPy, Pandas, Matplotlib
- PIL (Pillow)

## 🛠️ Installation

1. **Install Dependencies:**
```bash
pip install -r requirements_rag.txt
```

2. **Run the Application:**
```bash
python run_rag_app.py
```

3. **Access the Application:**
Open your browser and go to: `http://localhost:8502`

## 📁 Project Structure
```
waferintel/
├── run_rag_app.py              # Main launcher script
├── main_app_rag.py              # Main Streamlit application
├── rag_wafer_assistant.py       # AI assistant module
├── requirements_rag.txt         # Python dependencies
├── models/                      # Trained ML models
│   ├── wafer_model.h5          # WM811K CNN model
│   ├── external_wafer_model.h5 # MixedWM38 CNN model
│   ├── wafer_class_names.npy   # WM811K class labels
│   ├── external_wafer_classes.npy # MixedWM38 class labels
│   └── *.json                  # Model metrics
└── README.md                    # This file
```

## 🔧 Usage Instructions

### 1. Defect Detection
- Upload wafer images (PNG, JPG, JPEG)
- Select model (WM811K or MixedWM38)
- View AI-powered defect classification
- Get confidence scores and analysis

### 2. Model Insights
- Explore model performance metrics
- Compare dataset characteristics
- View accuracy metrics and confusion matrices
- Analyze model architecture details

### 3. AI Assistant
- Ask questions about semiconductor manufacturing
- Get defect-specific explanations
- Receive troubleshooting guidance
- Access expert knowledge base

## 🎯 Supported Defect Types
- **Center**: Central wafer defects
- **Donut**: Ring-shaped defects
- **Edge-Loc**: Edge-localized defects
- **Edge-Ring**: Edge ring defects
- **Loc**: Localized spot defects
- **Near-full**: Large area defects
- **Random**: Random pattern defects
- **Scratch**: Linear scratch defects
- **None**: No defects detected

## 📊 Model Information
- **WM811K Model**: 9-class single-label classification (45x45 images)
- **MixedWM38 Model**: 8-class multi-label classification (32x32 images)

## 🔒 Troubleshooting
- **Port Issues**: If port 8502 is busy, the app will automatically try alternative ports
- **Model Loading**: Ensure all model files are present in the `models/` directory
- **Dependencies**: Install all requirements from `requirements_rag.txt`

## 📞 Support
For technical issues or questions, please refer to the AI Assistant within the application or check the console output for error messages.

## 🏆 Project Achievement
This project demonstrates advanced AI applications in semiconductor manufacturing, combining:
- Deep learning for defect detection
- Real-time web interface
- Expert system integration
- Professional UI/UX design

---

**WaferIntel - Intelligent Wafer Defect Detection**  
*AI-Powered Semiconductor Quality Control* 🚀
