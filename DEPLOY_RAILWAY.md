# Deploy WaferIntel to Railway

## 🚀 Professional Cloud Hosting

### Why Railway?
- **Free Tier**: $5/month credit (enough for development)
- **Custom Domains**: Professional branding
- **Easy Deployment**: GitHub integration
- **SSL Certificate**: Automatic HTTPS
- **Global CDN**: Fast worldwide access

### Prerequisites
- Railway account (free at https://railway.app)
- GitHub repository with WaferIntel code

### Deployment Steps

1. **Go to Railway**: https://railway.app/new
2. **Connect GitHub**: Authorize Railway to access your repositories
3. **Select Repository**: Choose `waferintel` repository
4. **Configure Service**:
   - Service Type: `Web Service`
   - Build Command: `pip install -r requirements_rag.txt && python run_rag_app.py`
   - Port: `8502`
   - Health Check Path: `/`
5. **Deploy**: Click "Deploy Now"

### Environment Variables
Set these in Railway dashboard:
```bash
PORT=8502
STREAMLIT_SERVER_PORT=8502
STREAMLIT_SERVER_HEADLESS=false
```

### Your App URL
After deployment:
```
https://waferintel-production.up.railway.app
```

### Custom Domain
1. Go to Railway dashboard
2. Select your service
3. Click "Settings" → "Domains"
4. Add custom domain: `waferintel.yourdomain.com`

### Monitoring
- **Logs**: Real-time application logs
- **Metrics**: CPU, memory, and network usage
- **Alerts**: Email notifications for errors
- **Backups**: Automatic snapshots

### Scaling Options
- **Free Tier**: Shared resources
- **Hobby Tier**: $5/month - Dedicated resources
- **Pro Tier**: $20/month - High performance
- **Enterprise**: Custom pricing

---

**Professional hosting with Railway!** 🚀
