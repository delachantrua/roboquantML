# BingX Trading Bots

## 🤖 Bots Included:

### 1. BingX Market Maker Bot with ML
- **File**: `bingx_market_maker.py`
- **Features**: Avellaneda-Stoikov strategy + Machine Learning
- **ML Capabilities**: Forward testing, real-time predictions
- **Risk Management**: Inventory control, position sizing

### 2. Telegram Signal Bot
- **File**: `auto_telegram_bot.py`
- **Features**: Automatic signal reading from Telegram
- **Trading**: Auto-executes signals with TP/SL
- **Risk**: Low leverage (10x), small amounts ($20)

### 3. Reality Check Script
- **File**: `check_bingx_reality.py`
- **Purpose**: Verify actual BingX account status
- **Shows**: Balance, orders, positions, trades

## 🚀 Setup Instructions:

### 1. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys:
Edit each bot file and add your BingX API credentials:
```python
API_KEY = "YOUR_BINGX_API_KEY"
API_SECRET = "YOUR_BINGX_API_SECRET"
```

### 3. Run Bots:
```bash
# Market Maker Bot
python bingx_market_maker.py
```

## ⚠️ Important Notes:

- **Always test in sandbox mode first**
- **Start with small amounts**
- **Monitor your trades closely**
- **These bots use real money when not in sandbox**

## 🔧 Configuration:

- **Leverage**: 5x (Market Maker)
- **Order Size**: $0.01 ETH (Market Maker)
- **Risk Management**: Built-in inventory limits
- **ML Features**: Real-time forward testing

## 📊 Features:

✅ **Market Making**: Professional Avellaneda-Stoikov strategy  
✅ **Machine Learning**: Forward testing (better than backtesting)  
✅ **Risk Management**: Position sizing, inventory control  
✅ **Real-time Monitoring**: Live P&L, order status  
✅ **BingX Optimized**: Hedge mode, futures trading  

## 🆘 Support:

If you encounter issues:
1. Check your API credentials
2. Verify BingX account permissions
3. Test with small amounts first
4. Monitor the bot logs

---
*Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
