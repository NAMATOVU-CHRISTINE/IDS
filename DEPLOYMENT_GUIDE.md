# 🚀 Deployment Guide - IDS System

## Overview
This guide explains how to deploy the Intrusion Detection System to production.

## Architecture

```
GitHub Repository
       ↓
   ┌───┴───┐
   ↓       ↓
Render    Streamlit Cloud
(API)     (Dashboard)
```

## Prerequisites

- GitHub account
- Render account (free tier)
- Streamlit Cloud account (free tier)
- Git installed locally

## Part 1: Deploy API to Render

### Step 1: Prepare Repository

1. Ensure these files exist in your repo:
   - `api_render.py` - Flask API code
   - `requirements_api.txt` - API dependencies
   - `train_and_save_model.py` - Model training script
   - `build.sh` - Build script
   - `input/nslkdd/KDDTrain+.txt` - Training data

2. Push to GitHub:
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### Step 2: Create Render Web Service

1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `ids-api` (or your choice)
   - **Environment**: Python 3
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn api_render:app`
   - **Instance Type**: Free

5. Click "Create Web Service"

### Step 3: Wait for Deployment

- Build takes 5-10 minutes
- Trains Random Forest model during build
- Check logs for "Model training complete!"

### Step 4: Test API

```bash
# Health check
curl https://your-api-url.onrender.com/health

# Get metrics
curl https://your-api-url.onrender.com/metrics

# Test prediction
curl -X POST https://your-api-url.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"src_bytes": 181, "dst_bytes": 5450, "protocol_type": "tcp", "service": "http", "flag": "SF"}'
```

## Part 2: Deploy Dashboard to Streamlit Cloud

### Step 1: Prepare Dashboard Files

Ensure these files exist:
- `streamlit_app.py` - Dashboard code
- `requirements.txt` - Dashboard dependencies

### Step 2: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your GitHub repository
4. Configure:
   - **Branch**: main
   - **Main file path**: streamlit_app.py
   - **Python version**: 3.9+

5. Click "Deploy"

### Step 3: Configure API Connection

In `streamlit_app.py`, update:
```python
API_URL = "https://your-api-url.onrender.com"
```

Push changes:
```bash
git add streamlit_app.py
git commit -m "Update API URL"
git push origin main
```

Streamlit Cloud will auto-redeploy.

## Part 3: Verify Deployment

### Test Complete System

1. Open dashboard URL
2. Go to "Live Detection" page
3. Enter test traffic data:
   - Source Bytes: 181
   - Destination Bytes: 5450
   - Protocol: tcp
   - Service: http
   - Flag: SF

4. Click "Analyze Traffic"
5. Verify prediction appears

## Troubleshooting

### API Issues

**Problem**: Model not loading
```bash
# Check Render logs
# Look for "Model loaded successfully"
```

**Solution**: Ensure `build.sh` runs `train_and_save_model.py`

**Problem**: Prediction fails
```bash
# Check preprocessing matches training
# Verify feature names match
```

### Dashboard Issues

**Problem**: API connection timeout
```bash
# Increase timeout in streamlit_app.py
timeout=15  # or higher
```

**Problem**: Dashboard not updating
```bash
# Force redeploy on Streamlit Cloud
# Or push a small change to trigger rebuild
```

## Monitoring

### Check API Health

```bash
# Automated health check
watch -n 30 'curl https://your-api-url.onrender.com/health'
```

### View Logs

- **Render**: Dashboard → Your Service → Logs
- **Streamlit**: Cloud dashboard → App → Logs

## Updating the System

### Update Model

1. Modify `train_and_save_model.py`
2. Push to GitHub
3. Render auto-redeploys

### Update Dashboard

1. Modify `streamlit_app.py`
2. Push to GitHub
3. Streamlit Cloud auto-redeploys

### Update API Logic

1. Modify `api_render.py`
2. Push to GitHub
3. Render auto-redeploys

## Performance Tips

### Render Free Tier

- Spins down after 15 min inactivity
- First request after spin-down takes 30-60 seconds
- Upgrade to paid tier for always-on

### Streamlit Cloud

- Free tier has resource limits
- Optimize large dataframes
- Cache expensive computations

## Security

### API Security (Optional)

Add API key authentication:

```python
# In api_render.py
API_KEY = os.environ.get('API_KEY')

@app.before_request
def check_api_key():
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401
```

Set in Render:
- Dashboard → Environment → Add Variable
- Key: `API_KEY`
- Value: your-secret-key

## Cost

- **Render Free Tier**: $0/month
  - 750 hours/month
  - Spins down after inactivity
  
- **Streamlit Cloud Free**: $0/month
  - 1 private app
  - Unlimited public apps

## Support

- Render Docs: https://render.com/docs
- Streamlit Docs: https://docs.streamlit.io
- GitHub Issues: Your repo issues page

---

**Last Updated**: November 14, 2025
