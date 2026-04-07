# Deploy WaferIntel to Streamlit Cloud

## 🚀 One-Click Deployment

### Prerequisites
- Streamlit account (free at https://streamlit.io)
- GitHub repository with your code

### Steps to Deploy

1. **Go to Streamlit Cloud**: https://share.streamlit.io
2. **Connect GitHub**: Click "Connect with GitHub"
3. **Select Repository**: Choose `waferintel` repository
4. **Configure App**: 
   - Main file path: `main_app_rag.py`
   - Python version: 3.8+
   - Requirements file: `requirements_rag.txt`
5. **Deploy**: Click "Deploy"

### Your App URL
After deployment, your app will be available at:
```
https://yourusername-waferintel.streamlit.app
```

### Features Available on Streamlit Cloud
- ✅ **Free hosting** for public repositories
- ✅ **Automatic deployments** on git push
- ✅ **Custom domain** support
- ✅ **Team collaboration** features
- ✅ **Performance monitoring** built-in

### Advanced Configuration
```yaml
# .streamlit/config.toml (optional)
[theme]
primaryColor = "#6366f1"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"

[server]
port = 8502
headless = true
```

### Troubleshooting
If deployment fails:
1. Check `requirements_rag.txt` has all dependencies
2. Verify `main_app_rag.py` is the main file
3. Ensure no syntax errors in Python files
4. Check Streamlit Cloud status page

### Benefits of Streamlit Cloud
- **Zero Infrastructure**: No server management needed
- **Global CDN**: Fast loading worldwide
- **SSL Certificate**: Automatic HTTPS
- **Custom Domain**: Professional branding
- **Analytics**: Built-in usage statistics

---

**Deploy in minutes, not hours!** 🚀
