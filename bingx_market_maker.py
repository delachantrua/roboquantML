#!/usr/bin/env python3
"""
Standalone BingX Market Making Bot - Avellaneda-Stoikov Strategy + ML Forward Testing
© 2025 - Professional Cryptocurrency Trading Solutions
Adapted from Bybit bot for BingX compatibility with Machine Learning

INSTRUCTIONS:
1. Edit the API_KEY and API_SECRET below with your BingX credentials
2. Adjust trading parameters if needed (or leave defaults)
3. Run: python bingx_market_maker.py
"""

import ccxt
import time
import math
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import statistics
import logging
from typing import Dict, Tuple, Optional, Any
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    print("⚠️  ML libraries not available. Install with: pip install scikit-learn pandas numpy")
    ML_AVAILABLE = False

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# BingX API Credentials (REQUIRED - Get from https://bingx.com/)
API_KEY = "YOUR_BINGX_API_KEY_HERE"  # Replace with your actual API key
API_SECRET = "YOUR_BINGX_API_SECRET_HERE"  # Replace with your actual API secret
SANDBOX_MODE = True  # Set to False for live trading

# Trading Configuration
SYMBOL = "ETH/USDT:USDT"  # ETH perpetual futures
LEVERAGE = 5  # Leverage multiplier
ORDER_SIZE_FIXED = 0.01  # Fixed order size in ETH (matches server)
ORDER_SIZE_PERCENT = 0.02  # Fallback percentage if fixed size fails

# Avellaneda-Stoikov Parameters (Exact Server Match)
GAMMA = 0.01  # Risk aversion parameter (γ) - Controls spread width
K = 5.0  # Market impact parameter (k) - For fill intensity modeling  
ALPHA = 0.001  # Inventory penalty parameter (α) - Separate from gamma
TIME_HORIZON = 0.1  # Time horizon in hours (6 minutes) - rolling calculation
SIGMA_LOOKBACK = 50  # Price history length for volatility (matches server)
UPDATE_FREQUENCY = 1.0  # Update quotes every 1 second (ultra aggressive)

# Risk Management (Server-tuned)
MAX_INVENTORY_USD = 200.0  # Maximum inventory in USD

# Machine Learning Configuration
ML_ENABLED = True  # Enable/disable ML features
ML_LOOKBACK = 1000  # Number of data points for ML training
ML_UPDATE_FREQUENCY = 60  # Update ML model every 60 seconds
ML_FORWARD_TESTING = True  # Enable forward testing mode
ML_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for ML signals
ML_FEATURE_COUNT = 20  # Number of features to select

# ============================================================================
# BOT CODE - NO NEED TO EDIT BELOW THIS LINE
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bingx_market_maker_ml.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BingXMarketMaker:
    """Standalone BingX market maker with ML forward testing"""
    
    def __init__(self):
        """Initialize the market maker with ML capabilities"""
        self.exchange = None
        self.symbol = SYMBOL
        self.price_history = deque(maxlen=max(SIGMA_LOOKBACK, ML_LOOKBACK))
        self.volume_history = deque(maxlen=ML_LOOKBACK)
        self.inventory = 0
        self.pnl = 0
        self.trades_count = 0
        self.current_orders = {'bid': None, 'ask': None}
        self.volatility = 0.01
        self.running = False
        self.last_trade_check = 0
        
        # ML Components
        self.ml_model = None
        self.ml_scaler = None
        self.ml_feature_selector = None
        self.ml_trained = False
        self.ml_last_update = 0
        self.ml_predictions = deque(maxlen=100)
        self.ml_accuracy_history = deque(maxlen=50)
        self.ml_forward_test_results = []
        
        # Timing - use strategy start time instead of wall clock (matches server)
        self.start_time = time.time()
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate the hardcoded configuration"""
        if not API_KEY or API_KEY == "YOUR API KEY" or "YOUR" in API_KEY:
            logger.error("Please set your actual BingX API_KEY in the configuration section")
            sys.exit(1)
            
        if not API_SECRET or API_SECRET == "YOUR API SECRET" or "YOUR" in API_SECRET:
            logger.error("Please set your actual BingX API_SECRET in the configuration section")
            sys.exit(1)
            
        logger.info("Configuration validated successfully")
    
    def initialize_exchange(self) -> None:
        """Initialize the BingX exchange connection"""
        exchange_config = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap'  # For perpetual contracts
            }
        }
        
        # Set sandbox mode if enabled
        if SANDBOX_MODE:
            exchange_config['sandbox'] = True
            logger.info("Running in SANDBOX/TESTNET mode")
        
        # Initialize BingX exchange
        self.exchange = ccxt.bingx(exchange_config)
        
        # Load markets
        try:
            self.exchange.load_markets()
            logger.info("Successfully connected to BingX")
        except Exception as e:
            logger.error(f"Failed to connect to BingX: {e}")
            raise
    
    def validate_symbol(self) -> None:
        """Validate and set the trading symbol"""
        if self.symbol not in self.exchange.markets:
            available_symbols = [s for s in self.exchange.markets.keys() if 'USDT' in s]
            logger.error(f"Symbol {self.symbol} not found. Available symbols: {available_symbols[:10]}...")
            raise ValueError(f"Invalid symbol: {self.symbol}")
        
        market = self.exchange.markets[self.symbol]
        logger.info(f"Trading symbol: {self.symbol}")
        logger.info(f"Min order size: {market.get('limits', {}).get('amount', {}).get('min', 'Unknown')}")
        logger.info(f"Price precision: {market.get('precision', {}).get('price', 'Unknown')}")
    
    def set_leverage(self) -> None:
        """Set leverage for the trading pair"""
        try:
            if hasattr(self.exchange, 'set_leverage'):
                # BingX hedge mode requires separate leverage for LONG and SHORT
                self.exchange.set_leverage(LEVERAGE, self.symbol, params={'side': 'LONG'})
                self.exchange.set_leverage(LEVERAGE, self.symbol, params={'side': 'SHORT'})
                logger.info(f"Leverage set to {LEVERAGE}x for both LONG and SHORT")
        except Exception as e:
            logger.warning(f"Could not set leverage: {e}")
    
    def calculate_ml_features(self) -> np.ndarray:
        """Calculate ML features from price and volume data"""
        try:
            if len(self.price_history) < 50:
                return None
            
            prices = np.array(list(self.price_history))
            volumes = np.array(list(self.volume_history)) if self.volume_history else np.ones_like(prices)
            
            # Price-based features
            returns = np.diff(np.log(prices))
            price_change = np.diff(prices)
            price_momentum = prices[-1] / prices[-20] - 1 if len(prices) >= 20 else 0
            
            # Volatility features
            volatility_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0
            volatility_20 = np.std(returns[-20:]) if len(returns) >= 20 else 0
            volatility_50 = np.std(returns[-50:]) if len(returns) >= 50 else 0
            
            # Moving averages
            ma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
            ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
            
            # RSI-like features
            gains = np.sum(returns[returns > 0][-14:]) if len(returns) >= 14 else 0
            losses = np.abs(np.sum(returns[returns < 0][-14:])) if len(returns) >= 14 else 0
            rsi = gains / (gains + losses) if (gains + losses) > 0 else 0.5
            
            # Volume features
            volume_ma = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1
            volume_ratio = volumes[-1] / volume_ma if volume_ma > 0 else 1
            
            # Bollinger Bands
            bb_upper = ma_20 + 2 * volatility_20 * ma_20
            bb_lower = ma_20 - 2 * volatility_20 * ma_20
            bb_position = (prices[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            
            # MACD-like features
            macd = ma_5 - ma_20
            macd_signal = ma_5 - ma_50
            
            # Price patterns
            higher_highs = np.sum(np.diff(prices[-10:]) > 0) if len(prices) >= 10 else 0
            lower_lows = np.sum(np.diff(prices[-10:]) < 0) if len(prices) >= 10 else 0
            
            # Combine all features
            features = np.array([
                prices[-1], prices[-5], prices[-10], prices[-20],  # Price levels
                returns[-1], returns[-5], returns[-10],  # Recent returns
                volatility_5, volatility_20, volatility_50,  # Volatility
                ma_5, ma_20, ma_50,  # Moving averages
                rsi, bb_position,  # Technical indicators
                macd, macd_signal,  # MACD
                volume_ratio,  # Volume
                higher_highs, lower_lows,  # Patterns
                price_momentum  # Momentum
            ])
            
            return features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error calculating ML features: {e}")
            return None
    
    def create_ml_labels(self, lookahead: int = 10) -> np.ndarray:
        """Create labels for ML training (future price direction)"""
        try:
            if len(self.price_history) < lookahead + 1:
                return None
            
            prices = np.array(list(self.price_history))
            future_returns = []
            
            for i in range(len(prices) - lookahead):
                future_return = (prices[i + lookahead] - prices[i]) / prices[i]
                future_returns.append(future_return)
            
            # Create binary labels: 1 for positive return, 0 for negative
            labels = np.array([1 if ret > 0 else 0 for ret in future_returns])
            
            return labels
            
        except Exception as e:
            logger.error(f"Error creating ML labels: {e}")
            return None
    
    def train_ml_model(self) -> None:
        """Train the machine learning model"""
        try:
            if not ML_ENABLED or len(self.price_history) < ML_LOOKBACK:
                return
            
            logger.info("🧠 Training ML model...")
            
            # Prepare features and labels
            features_list = []
            labels_list = []
            
            # Create training data from historical data
            for i in range(50, len(self.price_history) - 10):
                # Get features for this point in time
                temp_prices = list(self.price_history)[:i+1]
                temp_volumes = list(self.volume_history)[:i+1] if self.volume_history else [1] * len(temp_prices)
                
                # Calculate features
                self.price_history = deque(temp_prices, maxlen=ML_LOOKBACK)
                self.volume_history = deque(temp_volumes, maxlen=ML_LOOKBACK)
                
                features = self.calculate_ml_features()
                if features is not None:
                    features_list.append(features.flatten())
                    
                    # Calculate future return
                    if i + 10 < len(temp_prices):
                        future_return = (temp_prices[i + 10] - temp_prices[i]) / temp_prices[i]
                        labels_list.append(1 if future_return > 0 else 0)
            
            if len(features_list) < 100:
                logger.warning("Insufficient data for ML training")
                return
            
            # Convert to numpy arrays
            X = np.array(features_list)
            y = np.array(labels_list)
            
            # Feature selection
            self.ml_feature_selector = SelectKBest(score_func=f_regression, k=ML_FEATURE_COUNT)
            X_selected = self.ml_feature_selector.fit_transform(X, y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.ml_scaler = StandardScaler()
            X_train_scaled = self.ml_scaler.fit_transform(X_train)
            X_test_scaled = self.ml_scaler.transform(X_test)
            
            # Train model
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.ml_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.ml_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            logger.info(f"✅ ML Model trained successfully!")
            logger.info(f"📊 Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            self.ml_trained = True
            self.ml_last_update = time.time()
            
            # Save model
            self.save_ml_model()
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
    
    def predict_with_ml(self, features: np.ndarray) -> Tuple[float, float]:
        """Make prediction using trained ML model"""
        try:
            if not self.ml_trained or self.ml_model is None:
                return 0.5, 0.0
            
            # Select features
            features_selected = self.ml_feature_selector.transform(features)
            
            # Scale features
            features_scaled = self.ml_scaler.transform(features_selected)
            
            # Make prediction
            prediction = self.ml_model.predict_proba(features_scaled)[0]
            confidence = max(prediction)
            
            # Return probability of positive return and confidence
            return prediction[1], confidence
            
        except Exception as e:
            logger.error(f"Error making ML prediction: {e}")
            return 0.5, 0.0
    
    def save_ml_model(self) -> None:
        """Save the trained ML model"""
        try:
            if self.ml_model is not None:
                model_data = {
                    'model': self.ml_model,
                    'scaler': self.ml_scaler,
                    'feature_selector': self.ml_feature_selector,
                    'timestamp': time.time()
                }
                
                with open('ml_model.pkl', 'wb') as f:
                    pickle.dump(model_data, f)
                
                logger.info("💾 ML model saved successfully")
                
        except Exception as e:
            logger.error(f"Error saving ML model: {e}")
    
    def load_ml_model(self) -> None:
        """Load a previously trained ML model"""
        try:
            if os.path.exists('ml_model.pkl'):
                with open('ml_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                self.ml_model = model_data['model']
                self.ml_scaler = model_data['scaler']
                self.ml_feature_selector = model_data['feature_selector']
                self.ml_trained = True
                
                logger.info("📂 ML model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
    
    def forward_test_ml(self, current_price: float, prediction: float, confidence: float) -> None:
        """Perform forward testing of ML predictions"""
        try:
            if not ML_FORWARD_TESTING:
                return
            
            # Store prediction for future validation
            forward_test_entry = {
                'timestamp': time.time(),
                'price': current_price,
                'prediction': prediction,
                'confidence': confidence,
                'actual_return': None,
                'validated': False
            }
            
            self.ml_forward_test_results.append(forward_test_entry)
            
            # Validate previous predictions
            self.validate_ml_predictions(current_price)
            
            # Calculate forward testing accuracy
            if len(self.ml_forward_test_results) > 10:
                validated_predictions = [r for r in self.ml_forward_test_results if r['validated']]
                if validated_predictions:
                    correct_predictions = sum(1 for r in validated_predictions if 
                                           (r['prediction'] > 0.5 and r['actual_return'] > 0) or
                                           (r['prediction'] < 0.5 and r['actual_return'] < 0))
                    
                    accuracy = correct_predictions / len(validated_predictions)
                    self.ml_accuracy_history.append(accuracy)
                    
                    avg_accuracy = np.mean(self.ml_accuracy_history) if self.ml_accuracy_history else 0
                    logger.info(f"🔮 ML Forward Testing: Recent Accuracy: {accuracy:.3f}, Avg: {avg_accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error in forward testing: {e}")
    
    def validate_ml_predictions(self, current_price: float) -> None:
        """Validate previous ML predictions"""
        try:
            current_time = time.time()
            
            for entry in self.ml_forward_test_results:
                if entry['validated']:
                    continue
                
                # Check if enough time has passed (10 periods)
                time_diff = current_time - entry['timestamp']
                if time_diff >= 10:  # 10 seconds
                    # Calculate actual return
                    actual_return = (current_price - entry['price']) / entry['price']
                    entry['actual_return'] = actual_return
                    entry['validated'] = True
                    
                    # Log validation result
                    prediction_correct = ((entry['prediction'] > 0.5 and actual_return > 0) or
                                        (entry['prediction'] < 0.5 and actual_return < 0))
                    
                    status = "✅" if prediction_correct else "❌"
                    logger.info(f"🔍 ML Validation {status}: Pred: {entry['prediction']:.3f}, "
                              f"Actual: {actual_return:.3f}, Conf: {entry['confidence']:.3f}")
            
        except Exception as e:
            logger.error(f"Error validating ML predictions: {e}")
    
    def calculate_volatility(self) -> float:
        """Calculate realized volatility from price history"""
        if len(self.price_history) < 2:
            return self.volatility
        
        returns = []
        for i in range(1, len(self.price_history)):
            ret = math.log(self.price_history[i] / self.price_history[i-1])
            returns.append(ret)
        
        if len(returns) > 1:
            self.volatility = statistics.stdev(returns) * math.sqrt(3600)
            self.volatility = max(self.volatility, 0.001)
        
        return self.volatility
    
    def calculate_reservation_price(self, mid_price: float) -> float:
        """Calculate reservation price with proper inventory penalty (matches server)"""
        sigma = self.calculate_volatility()
        time_remaining = self.get_time_remaining()
        
        # Correct A-S formula: r = m - α * q * σ² * T
        # Note: Using alpha (inventory penalty), not gamma (risk aversion)
        inventory_penalty = ALPHA * self.inventory * sigma**2 * time_remaining
        reservation_price = mid_price - inventory_penalty
        
        return reservation_price
    
    def get_time_remaining(self) -> float:
        """Get time remaining in current strategy horizon (matches server)"""
        # Use rolling time horizon from strategy start, not wall clock
        elapsed_hours = (time.time() - self.start_time) / 3600
        cycle_position = elapsed_hours % TIME_HORIZON
        time_remaining = TIME_HORIZON - cycle_position
        return max(time_remaining, 0.01)  # Minimum time remaining
    
    def calculate_optimal_spread(self, mid_price: float, ml_signal: float = 0.5) -> float:
        """Calculate optimal bid-ask spread using Avellaneda-Stoikov + ML adjustment"""
        sigma = self.calculate_volatility()
        time_remaining = self.get_time_remaining()
        
        # Base A-S spread
        risk_term = GAMMA * sigma**2 * time_remaining
        market_impact_term = (2 / GAMMA) * math.log(1 + GAMMA / K)
        base_spread = risk_term + market_impact_term
        
        # ML adjustment: widen spread if ML predicts high volatility
        if ML_ENABLED and self.ml_trained:
            ml_adjustment = 1.0 + (ml_signal - 0.5) * 0.5  # ±25% adjustment
            base_spread *= ml_adjustment
        
        # Server's constraints: minimum spread (wider than before)
        min_spread_bps = 2.0  # 2 basis points minimum
        min_spread = (min_spread_bps / 10000) * mid_price
        spread = max(base_spread * mid_price, min_spread)
        
        # Server's maximum spread constraint (CRITICAL!)
        max_spread_bps = 20.0  # 20 basis points maximum
        max_spread = (max_spread_bps / 10000) * mid_price
        spread = min(spread, max_spread)
        
        return spread
    
    def calculate_quote_prices(self, mid_price: float, ml_signal: float = 0.5) -> Tuple[float, float]:
        """Calculate optimal bid and ask prices with ML adjustment"""
        reservation_price = self.calculate_reservation_price(mid_price)
        spread = self.calculate_optimal_spread(mid_price, ml_signal)
        
        # Calculate base bid and ask around reservation price
        bid_price = reservation_price - spread / 2
        ask_price = reservation_price + spread / 2
        
        # ML bias: adjust prices based on ML prediction
        if ML_ENABLED and self.ml_trained:
            ml_bias = (ml_signal - 0.5) * spread * 0.1  # ±10% of spread
            bid_price += ml_bias
            ask_price += ml_bias
        
        # Server's exact quote distance constraint (CRITICAL!)
        min_spread_from_mid = mid_price * 0.0005  # 5 bps minimum from mid (not 10bps!)
        bid_price = min(bid_price, mid_price - min_spread_from_mid)
        ask_price = max(ask_price, mid_price + min_spread_from_mid)
        
        # Round to exchange precision
        market = self.exchange.markets[self.symbol]
        price_precision = market.get('precision', {}).get('price', 0.01)
        
        if isinstance(price_precision, int):
            bid_price = round(bid_price, price_precision)
            ask_price = round(ask_price, price_precision)
        else:
            # Handle tick size
            tick_size = float(price_precision)
            bid_price = round(bid_price / tick_size) * tick_size
            ask_price = round(ask_price / tick_size) * tick_size
        
        return bid_price, ask_price
    
    def calculate_position_size(self, price: float, ml_signal: float = 0.5) -> float:
        """Calculate position size based on configuration and ML confidence"""
        market = self.exchange.markets[self.symbol]
        min_size = market.get('limits', {}).get('amount', {}).get('min', 0.01)
        
        # Use fixed order size (matches server behavior)
        base_size = ORDER_SIZE_FIXED
        
        # ML adjustment: increase size if ML is confident
        if ML_ENABLED and self.ml_trained:
            ml_confidence = abs(ml_signal - 0.5) * 2  # 0 to 1 scale
            if ml_confidence > ML_CONFIDENCE_THRESHOLD:
                base_size *= 1.2  # 20% size increase for high confidence
        
        # Inventory adjustment - reduce size when inventory is high
        inventory_value = abs(self.inventory * price)
        
        if inventory_value > MAX_INVENTORY_USD * 0.7:
            size_multiplier = 0.5
        elif inventory_value > MAX_INVENTORY_USD * 0.5:
            size_multiplier = 0.75
        else:
            size_multiplier = 1.0
        
        size = base_size * size_multiplier
        
        # Round to exchange precision
        amount_precision = market.get('precision', {}).get('amount', 0.01)
        if isinstance(amount_precision, int):
            size = round(size, amount_precision)
        else:
            # Handle lot size
            lot_size = float(amount_precision)
            size = round(size / lot_size) * lot_size
        
        return max(size, min_size)
    
    def get_available_balance(self) -> float:
        """Get available balance in USDT"""
        try:
            balance = self.exchange.fetch_balance()
            return balance.get('USDT', {}).get('free', 0)
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0
    
    def cancel_all_orders(self) -> None:
        """Cancel all open orders"""
        try:
            if hasattr(self.exchange, 'cancel_all_orders'):
                self.exchange.cancel_all_orders(self.symbol)
            else:
                open_orders = self.exchange.fetch_open_orders(self.symbol)
                for order in open_orders:
                    self.exchange.cancel_order(order['id'], self.symbol)
            
            self.current_orders = {'bid': None, 'ask': None}
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    def place_orders(self, bid_price: float, ask_price: float, size: float) -> None:
        """Place bid and ask orders with BingX-specific parameters"""
        self.cancel_all_orders()
        
        try:
            # Place bid order with BingX futures parameters
            bid_order = self.exchange.create_limit_order(
                self.symbol, 'buy', size, bid_price, params={
                    'positionSide': 'LONG',
                    'marginMode': 'cross',
                    'timeInForce': 'GTC'
                }
            )
            self.current_orders['bid'] = bid_order
            logger.info(f"Bid placed: {size} @ {bid_price}")
        except Exception as e:
            logger.error(f"Error placing bid: {e}")
        
        try:
            # Place ask order with BingX futures parameters
            ask_order = self.exchange.create_limit_order(
                self.symbol, 'sell', size, ask_price, params={
                    'positionSide': 'SHORT',
                    'marginMode': 'cross',
                    'timeInForce': 'GTC'
                }
            )
            self.current_orders['ask'] = ask_order
            logger.info(f"Ask placed: {size} @ {ask_price}")
        except Exception as e:
            logger.error(f"Error placing ask: {e}")
    
    def update_inventory(self) -> None:
        """Update inventory from actual positions (BingX-optimized)"""
        try:
            # Only check positions every 30 seconds to avoid rate limits
            current_time = time.time()
            if current_time - self.last_trade_check < 30:
                return
            
            self.last_trade_check = current_time
            
            # Get actual positions instead of trade history
            positions = self.exchange.fetch_positions([self.symbol])
            total_inventory = 0
            
            for position in positions:
                if float(position.get('contracts', 0)) != 0:  # Only non-zero positions
                    if position.get('side') == 'long':
                        total_inventory += float(position.get('contracts', 0))
                    elif position.get('side') == 'short':
                        total_inventory -= float(position.get('contracts', 0))
                    
                    # Update PnL from actual position data
                    unrealized_pnl = float(position.get('unrealizedPnl', 0))
                    self.pnl = unrealized_pnl
            
            self.inventory = total_inventory
            
            # Get actual trade count from positions
            if positions:
                self.trades_count = len([p for p in positions if float(p.get('contracts', 0)) != 0])
            
            logger.info(f"📊 Real inventory updated: {self.inventory:.3f} ETH, PnL: ${self.pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating inventory: {e}")
    
    def display_status(self, mid_price: float, bid_price: float, ask_price: float, size: float, ml_signal: float = 0.5, ml_confidence: float = 0.0) -> None:
        """Display current bot status with ML information"""
        spread = ask_price - bid_price
        spread_bps = (spread / mid_price) * 10000
        balance = self.get_available_balance()
        inventory_value = self.inventory * mid_price
        
        # Use dynamic time remaining to match server
        time_remaining = self.get_time_remaining()
        
        print(f"\n{'='*80}")
        print(f"🚀 BINGX MARKET MAKER BOT + ML FORWARD TESTING")
        print(f"ETH: ${mid_price:.2f} | Spread: {spread_bps:.1f}bps | σ: {self.volatility:.3f} | T-rem: {time_remaining:.3f}h")
        print(f"Inventory: {self.inventory:.3f} ETH (${inventory_value:.2f}) | Target: 0")
        print(f"Quotes: ${bid_price:.2f} / ${ask_price:.2f} | Size: {size:.3f} ETH")
        print(f"Stats: {self.trades_count} positions | PnL: ${self.pnl:.2f}")
        print(f"Balance: ${balance:.2f} USDT | k: {K:.2f} | γ: {GAMMA:.3f}")
        
        # ML Status
        if ML_ENABLED and self.ml_trained:
            ml_direction = "🟢 BULL" if ml_signal > 0.5 else "🔴 BEAR" if ml_signal < 0.5 else "⚪ NEUTRAL"
            ml_confidence_pct = ml_confidence * 100
            print(f"🧠 ML Signal: {ml_signal:.3f} ({ml_direction}) | Confidence: {ml_confidence_pct:.1f}%")
            
            if self.ml_accuracy_history:
                recent_accuracy = np.mean(list(self.ml_accuracy_history)[-5:]) if self.ml_accuracy_history else 0
                print(f"📊 ML Forward Testing: Recent Accuracy: {recent_accuracy:.1%}")
        else:
            print(f"🧠 ML Status: {'Training...' if ML_ENABLED else 'Disabled'}")
        
        # Risk indicators
        inventory_percent = abs(inventory_value) / MAX_INVENTORY_USD * 100
        if inventory_percent > 70:
            print(f"⚠️  HIGH INVENTORY RISK: {inventory_percent:.1f}%")
        elif inventory_percent > 50:
            print(f"⚡ MEDIUM INVENTORY: {inventory_percent:.1f}%")
        else:
            print(f"✅ INVENTORY OK: {inventory_percent:.1f}%")
    
    def run(self) -> None:
        """Main bot loop with ML integration"""
        print("🚀 Starting BingX Market Maker Bot + ML Forward Testing")
        print(f"📋 Configuration:")
        print(f"   Symbol: {SYMBOL}")
        print(f"   Leverage: {LEVERAGE}x")
        print(f"   Order Size: {ORDER_SIZE_FIXED} ETH (fixed)")
        print(f"   Max Inventory: ${MAX_INVENTORY_USD}")
        print(f"   Update Frequency: {UPDATE_FREQUENCY}s")
        print(f"   Parameters: γ={GAMMA}, k={K}, T={TIME_HORIZON}h (rolling)")
        print(f"   Sandbox Mode: {SANDBOX_MODE}")
        print(f"   ML Enabled: {ML_ENABLED}")
        print(f"   ML Forward Testing: {ML_FORWARD_TESTING}")
        
        # Initialize exchange
        self.initialize_exchange()
        self.validate_symbol()
        self.set_leverage()
        
        # Initialize ML
        if ML_ENABLED and ML_AVAILABLE:
            self.load_ml_model()
            if not self.ml_trained:
                logger.info("🧠 ML model not found, will train when enough data is available")
        elif ML_ENABLED and not ML_AVAILABLE:
            logger.warning("🧠 ML features disabled - required libraries not available")
            logger.info("💡 Install ML libraries with: pip install scikit-learn pandas numpy")
        
        self.running = True
        
        logger.info("Bot started successfully!")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Fetch orderbook
                orderbook = self.exchange.fetch_order_book(self.symbol)
                if not orderbook['bids'] or not orderbook['asks']:
                    logger.warning("Empty orderbook, retrying...")
                    time.sleep(1)
                    continue
                
                # Calculate mid price
                best_bid = orderbook['bids'][0][0]
                best_ask = orderbook['asks'][0][0]
                mid_price = (best_bid + best_ask) / 2
                
                # Get volume data
                ticker = self.exchange.fetch_ticker(self.symbol)
                volume = ticker.get('quoteVolume', 1)
                
                # Update price and volume history
                self.price_history.append(mid_price)
                self.volume_history.append(volume)
                
                # Update inventory
                self.update_inventory()
                
                # ML Processing
                ml_signal = 0.5
                ml_confidence = 0.0
                
                if ML_ENABLED and ML_AVAILABLE:
                    # Train/update ML model periodically
                    if time.time() - self.ml_last_update > ML_UPDATE_FREQUENCY:
                        self.train_ml_model()
                    
                    # Make ML prediction if model is trained
                    if self.ml_trained:
                        features = self.calculate_ml_features()
                        if features is not None:
                            ml_signal, ml_confidence = self.predict_with_ml(features)
                            
                            # Forward testing
                            self.forward_test_ml(mid_price, ml_signal, ml_confidence)
                            
                            # Store prediction
                            self.ml_predictions.append({
                                'timestamp': time.time(),
                                'signal': ml_signal,
                                'confidence': ml_confidence,
                                'price': mid_price
                            })
                
                # Check risk limits
                inventory_value = abs(self.inventory * mid_price)
                
                if inventory_value > MAX_INVENTORY_USD:
                    logger.warning(f"🚨 INVENTORY LIMIT REACHED: ${inventory_value:.2f} > ${MAX_INVENTORY_USD}")
                    self.cancel_all_orders()
                    time.sleep(10)
                    continue
                
                # Calculate quotes with ML adjustment
                bid_price, ask_price = self.calculate_quote_prices(mid_price, ml_signal)
                size = self.calculate_position_size(mid_price, ml_signal)
                
                # Display status
                self.display_status(mid_price, bid_price, ask_price, size, ml_signal, ml_confidence)
                
                # Place orders
                self.place_orders(bid_price, ask_price, size)
                
                # Sleep until next update
                elapsed = time.time() - start_time
                sleep_time = max(0, UPDATE_FREQUENCY - elapsed)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)
        
        # Cleanup
        self.cancel_all_orders()
        if ML_ENABLED and ML_AVAILABLE:
            self.save_ml_model()
        logger.info("🛑 Bot stopped")
    
    def stop(self) -> None:
        """Stop the bot"""
        self.running = False


def main():
    """Main entry point"""
    print("=" * 70)
    print("🎯 BINGX MARKET MAKER BOT + ML FORWARD TESTING")
    print("📈 Avellaneda-Stoikov Strategy + Machine Learning")
    print("⚡ Ultra High-Frequency Trading with AI")
    print("🔮 Real-time Forward Testing (Better than Backtesting)")
    print("=" * 70)
    
    # Create and run bot
    bot = BingXMarketMaker()
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
