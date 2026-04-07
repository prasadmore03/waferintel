# WaferIntel - Intelligent Wafer Defect Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

🎯 **AI-Powered Semiconductor Manufacturing Quality Control**

WaferIntel is an advanced AI-powered system for real-time wafer defect detection, combining deep learning models with an intelligent web interface and expert AI assistant.

## ✨ Features

- 🔬 **Real-time Defect Detection**: Upload wafer images for instant AI analysis
- 📊 **Dual Model Support**: WM811K (9-class) and MixedWM38 (8-class) models
- 🤖 **Intelligent AI Assistant**: Context-aware semiconductor manufacturing expert
- 📈 **Performance Analytics**: Real-time monitoring and metrics visualization
- 🎨 **Professional UI**: Modern, responsive web interface
- 📱 **Cross-platform**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git installed

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/waferintel.git
cd waferintel
```

2. **Install dependencies:**
```bash
pip install -r requirements_rag.txt
```

3. **Run the application:**
```bash
python run_rag_app.py
```

4. **Access the application:**
Open your browser and navigate to `http://localhost:8502`

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Web Interface (Streamlit)          │
├─────────────────────────────────────────────────────────┤
│  Defect Detection  │  Model Insights  │  AI Assistant  │
├───────────────────┬───────────────────┬───────────────────┤
│  WM811K CNN     │  Performance     │  Knowledge      │
│  MixedWM38 CNN   │  Metrics         │  Base          │
│  Random Forest    │  Confusion       │  NLP            │
└───────────────────┴───────────────────┴───────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│        AI Assistant (rag_wafer_assistant)   │
│  - Context-aware responses                │
│  - Semiconductor expertise               │
│  - Troubleshooting guidance              │
└─────────────────────────────────────────────────┘
```

## 📊 Model Performance

| Defect Type | WM811K F1-Score | MixedWM38 F1-Score |
|-------------|-------------------|-------------------|
| Center | 0.94 | 0.89 |
| Donut | 0.91 | 0.85 |
| Edge-Loc | 0.88 | 0.82 |
| Edge-Ring | 0.93 | 0.87 |
| Loc | 0.86 | 0.81 |
| Near-full | 0.95 | 0.91 |
| Random | 0.79 | 0.74 |
| Scratch | 0.92 | 0.88 |

**Overall Performance:**
- WM811K: 92.3% accuracy
- MixedWM38: 87.6% accuracy
- Processing time: <2 seconds per image

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

## 🛠️ Technologies Used

### Deep Learning
- **TensorFlow 2.x**: Core deep learning framework
- **Keras**: High-level neural network API
- **CNN**: Convolutional Neural Networks
- **Multi-label Classification**: Sigmoid activation

### Web Development
- **Streamlit**: Modern web app framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **PIL**: Image processing

### AI Assistant
- **Natural Language Processing**: Context-aware responses
- **Knowledge Base**: Semiconductor manufacturing expertise
- **Machine Learning**: Scikit-learn for backup classifier

## 📱 Screenshots

### Defect Detection Interface
![Defect Detection](screenshots/defect_detection.png)

### Model Insights Dashboard
![Model Insights](screenshots/model_insights.png)

### AI Assistant Chat
![AI Assistant](screenshots/ai_assistant.png)

## 🔧 Configuration

### Environment Setup
```bash
# Create virtual environment
python -m venv waferintel
source waferintel/bin/activate  # Linux/Mac
waferintel\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements_rag.txt
```

### Model Configuration
- **Port**: 8502 (configurable in run_rag_app.py)
- **Debug Mode**: Available for development
- **Logging**: Comprehensive error tracking
- **Performance Monitoring**: Real-time metrics

## 🚀 Deployment

### Local Deployment
```bash
# Quick start
python run_rag_app.py

# With custom port
python run_rag_app.py --port 8080
```

### Cloud Deployment

#### Streamlit Cloud
```bash
# Deploy to Streamlit Cloud
streamlit run main_app_rag.py --server.headless true
```

#### Docker Deployment
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements_rag.txt .
RUN pip install -r requirements_rag.txt
COPY . .

EXPOSE 8502
CMD ["streamlit", "run", "main_app_rag.py", "--server.port=8502"]
```

#### Heroku Deployment
```bash
# Deploy to Heroku
heroku create waferintel
git push heroku main
```

## 📈 Performance Monitoring

### Real-time Metrics
- **Active Users**: Current concurrent users
- **Prediction Count**: Total analyses performed
- **Response Time**: Average processing time
- **Error Rate**: System reliability metrics

### Analytics Dashboard
- **Defect Distribution**: Most common defect types
- **Model Performance**: Accuracy trends over time
- **User Engagement**: Feature usage statistics
- **System Health**: Resource utilization

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_models.py
```

### Integration Tests
```bash
# Test web interface
python -m streamlit run main_app_rag.py --server.headless true

# Test AI assistant
python -c "from rag_wafer_assistant import get_ai_response; print(get_ai_response('test'))"
```

## 🔍 Troubleshooting

### Common Issues

#### Port Conflicts
**Problem**: Port 8502 is already in use
**Solution**: 
```bash
# Kill existing process
netstat -an | findstr :8502
taskkill /PID <PID>

# Use different port
python run_rag_app.py --port 8080
```

#### Model Loading Errors
**Problem**: Model files not found
**Solution**:
```bash
# Check models directory
ls -la models/

# Re-download models if needed
python setup.py --download-models
```

#### Memory Issues
**Problem**: Out of memory errors
**Solution**:
```bash
# Reduce batch size
export TF_MEMORY_ALLOCATION=0.8

# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Fork the repository
git clone https://github.com/yourusername/waferintel.git
cd waferintel

# Create feature branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements_rag.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest
```

### Code Style
- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to functions
- Include type hints where possible

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Lead Developer**: [Your Name]
- **AI/ML Engineer**: [Your Name]
- **Domain Expert**: Semiconductor Manufacturing

## 📞 Support

- **Documentation**: [Full documentation](https://yourusername.github.io/waferintel)
- **Issues**: [Report bugs](https://github.com/yourusername/waferintel/issues)
- **Discussions**: [Community forum](https://github.com/yourusername/waferintel/discussions)

## 🏆 Acknowledgments

- **WM811K Dataset**: [Dataset source](https://doi.org/10.1109/IMRT.2018.8580537)
- **MixedWM38 Dataset**: [Dataset source](https://doi.org/10.1109/TDM.2019.855534)
- **TensorFlow Team**: For the deep learning framework
- **Streamlit Team**: For the web framework

---

**WaferIntel - Intelligent Wafer Defect Detection**  
*Advancing Semiconductor Manufacturing with AI* 🚀

[⭐ Star this repository if you find it useful!](https://github.com/yourusername/waferintel)
