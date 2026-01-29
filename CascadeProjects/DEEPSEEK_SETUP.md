# DeepSeek API Setup Guide

## ✅ Configuration Complete!

I've successfully migrated your system from Groq to DeepSeek:

### Changes Made:
1. **API Endpoint**: Changed to `https://api.deepseek.com/v1/chat/completions`
2. **Environment Variable**: Now uses `DEEPSEEK_API_KEY`
3. **Default Model**: Set to `deepseek-chat`
4. **Error Messages**: Updated to reference DeepSeek

### Your API Key:
```
sk-02c2494a697848bca565233cf6fabc7e
```

## ⚠️ Current Issue: Insufficient Balance

The API key is working correctly, but DeepSeek is returning:
```
{"error":{"message":"Insufficient Balance","type":"unknown_error"}}
```

### Solutions:

1. **Add Credits to Your DeepSeek Account**:
   - Visit: https://platform.deepseek.com/
   - Log in with your account
   - Go to "Billing" or "Credits"
   - Add funds to your account

2. **Check API Key Validity**:
   - Verify this is the correct API key
   - Make sure it's not expired
   - Confirm it has the right permissions

3. **Get a New API Key** (if needed):
   - Go to https://platform.deepseek.com/api_keys
   - Generate a new API key
   - Replace it in `setup_deepseek.sh`

## How to Use (Once Balance is Added):

### Option 1: Using the setup script
```bash
source setup_deepseek.sh
python3 openrouter_prompting_system.py prompt --input "your question"
```

### Option 2: Set environment variable directly
```bash
export DEEPSEEK_API_KEY="sk-02c2494a697848bca565233cf6fabc7e"
python3 openrouter_prompting_system.py prompt --input "your question"
```

### Option 3: Chat mode
```bash
export DEEPSEEK_API_KEY="sk-02c2494a697848bca565233cf6fabc7e"
python3 openrouter_prompting_system.py chat
```

## Testing the Connection:

Once you've added credits, test with:
```bash
export DEEPSEEK_API_KEY="sk-02c2494a697848bca565233cf6fabc7e"
python3 openrouter_prompting_system.py prompt --input "Hello! Please introduce yourself."
```

## Files Modified:
- `openrouter_prompting_system.py` - Updated API configuration
- `setup_deepseek.sh` - New setup script with your API key
