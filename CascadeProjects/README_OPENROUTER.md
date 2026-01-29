# OpenRouter Prompting System with 4D Prompt Agent

A sophisticated prompting system that uses OpenRouter as the LLM provider and incorporates a 4D prompt agent framework for multi-dimensional context analysis.

## Features

- **OpenRouter Integration**: Connect to multiple LLMs through OpenRouter API
- **4D Prompt Agent**: Analyzes interactions across four dimensions:
  - **Context**: Conversation history and topics
  - **Intent**: User goals and patterns
  - **Behavior**: User interaction style and engagement
  - **Temporal**: Time-based context
- **User & System Prompts**: Full support for both user and system-level prompts
- **Conversation History**: Maintains context across interactions
- **Configurable**: Easy configuration via JSON file

## Setup

1. **Install dependencies**:
   ```bash
   pip install requests
   ```

2. **Get OpenRouter API Key**:
  - Sign up at https://openrouter.ai/
  - Get your API key from the dashboard
  - Add it to `openrouter_config.json`, pass via `--api-key`, or set it as an environment variable `OPENROUTER_API_KEY` to avoid keeping secrets in the repo.
  - Note: This project now includes a `.gitignore` entry for `openrouter_config.json` to help prevent accidental commits of your secret; prefer environment variables or other secret stores in production.

3. **Configure**:
   Edit `openrouter_config.json`:
   ```json
   {
     "api_key": "your-api-key-here",
     "model": "openai/gpt-4o",
     "system_prompt": "Your custom system prompt...",
     "temperature": 0.7,
     "max_tokens": 2000
   }
   ```

## Usage

### Single Prompt
```bash
python3 openrouter_prompting_system.py \
  --api-key YOUR_API_KEY \
  prompt \
  --input "What is machine learning?"
```

### Interactive Chat
```bash
python3 openrouter_prompting_system.py \
  --api-key YOUR_API_KEY \
  chat
```

### Save Transcripts
```bash
python3 openrouter_prompting_system.py \
  --api-key YOUR_API_KEY \
  --save-transcript ./chat_logs \
  chat
```

### Disable 4D Analysis
```bash
python3 openrouter_prompting_system.py \
  --api-key YOUR_API_KEY \
  --no-4d \
  prompt \
  --input "Simple question"
```

## 4D Prompt Agent

The 4D prompt agent automatically analyzes each interaction:

1. **Context Dimension**: Tracks conversation topics and history
2. **Intent Dimension**: Detects user intent (question, command, analysis, comparison)
3. **Behavior Dimension**: Analyzes input style (detailed/concise) and engagement level
4. **Temporal Dimension**: Considers time of day and temporal context

This analysis is included in the system prompt to help the LLM provide more contextualized responses.

## Available Models

OpenRouter supports many models. Popular options:
- `openai/gpt-4o` - GPT-4 Omni
- `openai/gpt-4-turbo` - GPT-4 Turbo
- `anthropic/claude-3.5-sonnet` - Claude 3.5 Sonnet
- `google/gemini-pro-1.5` - Gemini Pro 1.5
- `meta-llama/llama-3.1-70b-instruct` - Llama 3.1 70B

See https://openrouter.ai/models for the full list.

## Configuration Options

- `api_key`: Your OpenRouter API key
- `model`: Model identifier (e.g., "openai/gpt-4o")
- `system_prompt`: Base system prompt (4D analysis is appended)
- `temperature`: Sampling temperature (0.0-2.0)
- `max_tokens`: Maximum tokens in response
- `http_referer`: Referer header for OpenRouter
- `app_name`: Application name for OpenRouter

