# Setup Guide for Prompt Suggestions Feature

## 🔑 Required API Keys

You need to create a `.env` file in the project root with these keys:

```bash
# Create .env file
touch .env
```

Add these variables to your `.env` file:

```env
# OpenAI API Key (required for AI responses and embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone API Key (required for vector database)
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone Index Name (your vector database index)
PINECONE_INDEX_NAME=your_index_name_here

# Pinecone Environment (e.g., us-east-1-aws)
PINECONE_ENVIRONMENT=your_pinecone_environment_here
```

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

## 🧪 Test the Feature

1. **Start the server:**
   ```bash
   python main.py
   ```

2. **Open the frontend:**
   - Open `index.html` in your browser
   - Connect to a room (e.g., "test_room")
   - Start chatting

3. **You should see:**
   - AI responses to your questions
   - 3 clickable prompt suggestions after each response
   - Suggestions only during regular conversations (not quizzes)

## 🔍 How It Works

1. **Student asks a question** → AI responds
2. **System generates 3 contextual suggestions** using OpenAI
3. **Suggestions appear as clickable buttons**
4. **Student clicks any suggestion** to send it as their next question

## ⚠️ Important Notes

- The feature requires working OpenAI and Pinecone API keys
- Suggestions are generated using AI, so they'll be contextual and relevant
- If API keys are missing, the system will fall back to basic suggestions
- The feature only works during regular conversations, not during quizzes

## 🐛 Troubleshooting

If suggestions don't appear:
1. Check that your `.env` file has all required keys
2. Verify API keys are valid and have sufficient credits
3. Check the console for error messages
4. Make sure you're not in a quiz mode (suggestions are disabled during quizzes)
