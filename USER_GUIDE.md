# 📖 User Guide - Intrusion Detection System

## Welcome!

This guide will help you use the IDS Dashboard to detect network intrusions in real-time.

## Accessing the Dashboard

**URL**: https://namatovu-christine-kjcaxsyrtwm7t7vkqp9jat.streamlit.app

No login required - just open the link in your browser!

## Dashboard Pages

### 🏠 Home Page

**What you'll see:**
- System metrics (models trained, accuracy, dataset size)
- Model performance comparison chart
- Best model statistics (Random Forest - 99% accuracy)

**Use this page to:**
- Get an overview of the system
- See which ML model performs best
- Understand the system capabilities

### 🔍 Live Detection Page

**What it does:**
- Analyzes network traffic in real-time
- Predicts if traffic is Normal or Attack
- Shows confidence scores

**How to use:**

1. **Enter Traffic Parameters:**
   - **Duration**: Connection duration in seconds (usually 0 for quick connections)
   - **Source Bytes**: Data sent from source (e.g., 181)
   - **Destination Bytes**: Data received at destination (e.g., 5450)
   - **Protocol**: tcp, udp, or icmp
   - **Service**: http, ftp, smtp, telnet, ssh, or private
   - **Flag**: Connection status (SF = normal, S0 = connection attempt, REJ = rejected)
   - **Count**: Number of connections to same host
   - **Service Count**: Number of connections to same service
   - **SYN Error Rate**: Percentage of SYN errors (0.0 to 1.0)

2. **Click "Analyze Traffic"**

3. **View Results:**
   - ✅ **Normal Traffic**: Safe connection
   - ⚠️ **Attack Detected**: Suspicious activity with recommended actions

**Example Scenarios:**

**Normal Web Traffic:**
```
Duration: 0
Source Bytes: 181
Destination Bytes: 5450
Protocol: tcp
Service: http
Flag: SF
Count: 1
Service Count: 1
SYN Error Rate: 0.0
```

**Potential Attack:**
```
Duration: 0
Source Bytes: 10000
Destination Bytes: 0
Protocol: tcp
Service: private
Flag: S0
Count: 100
Service Count: 100
SYN Error Rate: 1.0
```

### 📊 Analytics Page

**What you'll see:**

**Tab 1: Model Comparison**
- Table comparing all 7 ML models
- Accuracy, Precision, Recall, F1-Score for each

**Tab 2: Confusion Matrix**
- Visual representation of Random Forest predictions
- True Positives, False Positives, True Negatives, False Negatives

**Use this page to:**
- Compare different ML algorithms
- Understand model performance
- See prediction accuracy breakdown

### ℹ️ About Page

**What you'll find:**
- Project overview
- Team members and roles
- Technologies used
- Dataset information
- GitHub repository link

## Understanding Results

### Confidence Score

- **80-90%**: Moderate confidence
- **90-95%**: High confidence
- **95-100%**: Very high confidence

### Attack Indicators

High-risk patterns:
- Large source bytes (> 5000)
- Zero destination bytes
- High SYN error rate (> 0.5)
- Many connections (count > 50)
- Connection flag S0 or REJ

### Recommended Actions

When attack is detected:
1. 🚫 **Block source IP** - Prevent further connections
2. 📝 **Log incident** - Record for analysis
3. 🔔 **Alert security team** - Notify administrators

## Common Use Cases

### 1. Monitor Web Server Traffic

Check HTTP connections:
- Protocol: tcp
- Service: http
- Flag: SF (successful)
- Normal bytes: 100-5000

### 2. Detect Port Scanning

Look for:
- High connection count
- Multiple services
- Flag: S0 or REJ
- Low bytes transferred

### 3. Identify DoS Attacks

Characteristics:
- Very high source bytes
- Many connections to same host
- High error rates
- Service: private or other

### 4. Check Email Server

Monitor SMTP traffic:
- Protocol: tcp
- Service: smtp
- Normal flag: SF
- Moderate byte counts

## Tips for Best Results

1. **Use Realistic Values**
   - Don't enter random numbers
   - Base on actual network traffic

2. **Check Multiple Scenarios**
   - Test different protocols
   - Try various services
   - Compare normal vs suspicious

3. **Understand Your Network**
   - Know typical traffic patterns
   - Identify baseline metrics
   - Recognize anomalies

4. **Review Analytics**
   - Check model performance
   - Understand accuracy metrics
   - Trust high-confidence predictions

## Troubleshooting

### "API request timed out"

**Cause**: API server is waking up (free tier spins down)

**Solution**: Wait 30-60 seconds and try again

### "Failed to get prediction"

**Cause**: API connection issue

**Solution**: 
- Check internet connection
- Refresh page
- Try again in a few minutes

### Unexpected Results

**Cause**: Unusual input values

**Solution**:
- Use realistic network values
- Check protocol/service combinations
- Review example scenarios above

## Technical Details

### Dataset
- **Name**: NSL-KDD
- **Samples**: 125,972 training records
- **Features**: 122 (after preprocessing)
- **Classes**: Normal, Attack

### Model
- **Algorithm**: Random Forest
- **Accuracy**: 99.99%
- **Precision**: 99.93%
- **Recall**: 99.81%

### Attack Types Detected
1. **DoS** (Denial of Service)
2. **Probe** (Port scanning)
3. **R2L** (Remote to Local)
4. **U2R** (User to Root)

## Need Help?

- **GitHub**: https://github.com/NAMATOVU-CHRISTINE/IDS
- **Issues**: Report bugs on GitHub Issues
- **Contact**: Through MUST Computer Science Department

## Glossary

- **Protocol**: Communication method (TCP, UDP, ICMP)
- **Service**: Application type (HTTP, FTP, SMTP)
- **Flag**: Connection status indicator
- **SYN Error**: Failed connection attempt
- **Source Bytes**: Data sent from client
- **Destination Bytes**: Data received from server
- **Confidence**: Model's certainty in prediction
- **Precision**: Accuracy of attack predictions
- **Recall**: Percentage of attacks detected

---

**Version**: 1.0  
**Last Updated**: November 14, 2025  
**Developed by**: MUST Computer Science Team
