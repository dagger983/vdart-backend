require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const { GoogleGenAI } = require("@google/genai");
const rateLimit = require('express-rate-limit');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (GEMINI_API_KEY && !GEMINI_API_KEY.startsWith("AIzaSy") && !GEMINI_API_KEY.startsWith("AQ.")) {
  console.error("\n❌ ERROR: Invalid GEMINI_API_KEY format in .env file.");
  console.error("The key provided does not look like a valid Gemini API key (should start with 'AIzaSy' or 'AQ.').");
  console.error("Please get a valid API key from https://aistudio.google.com/app/apikey\n");
}

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

// ── IN-MEMORY CACHE & LOCKS (Replaced Redis) ───────────
const cacheStore = new Map();
const requestLocks = new Set();

// ── CACHING FUNCTIONS ──────────────────────────────────
async function getCachedResponse(prompt) {
  const hash = crypto.createHash('sha256').update(prompt.toLowerCase().trim()).digest('hex');
  const cached = cacheStore.get(`cache:${hash}`);
  return cached ? cached : null;
}
async function setCachedResponse(prompt, responseText) {
  const hash = crypto.createHash('sha256').update(prompt.toLowerCase().trim()).digest('hex');
  cacheStore.set(`cache:${hash}`, responseText);
  // Auto-expire after 24 hours (86400 seconds)
  setTimeout(() => cacheStore.delete(`cache:${hash}`), 86400 * 1000);
}

// ── MESSAGE GENERATION ─────────────────────────────────
async function generateChatResponse(jobData) {
  if (GEMINI_API_KEY && !GEMINI_API_KEY.startsWith("AIzaSy") && !GEMINI_API_KEY.startsWith("AQ.")) {
    throw new Error("Invalid GEMINI_API_KEY. Gemini API keys must start with 'AIzaSy' or 'AQ.'. Please check your .env file and get a valid key from https://aistudio.google.com/app/apikey.");
  }
  const { prompt, model = "gemini-2.5-flash", messages, systemInstruction } = jobData;

  if (messages) {
    // Format messages into Gemini API format: { role, parts: [{ text }] }
    const formattedContents = messages.map((msg) => {
      const role = (msg.role === "assistant" || msg.role === "model") ? "model" : "user";
      const text = msg.content || msg.text || "";
      return {
        role,
        parts: [{ text }]
      };
    });

    const response = await ai.models.generateContent({
      model: model,
      contents: formattedContents,
      config: {
        systemInstruction: systemInstruction,
        tools: [{ googleSearch: {} }]
      }
    });
    return response.text;
  } else {
    // SDK format for /testing-ai
    const response = await ai.models.generateContent({
      model: model,
      contents: prompt,
      config: {
        systemInstruction: systemInstruction,
        tools: [{ googleSearch: {} }]
      }
    });
    return response.text;
  }
}

// ── MIDDLEWARE ─────────────────────────────────────────
const apiLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 10,
  message: { error: 'Too many requests, please try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
});

const preventParallelRequests = async (req, res, next) => {
  const userId = req.ip;

  if (requestLocks.has(userId)) {
    return res.status(429).json({ error: 'Please wait for your previous request to finish.' });
  }

  requestLocks.add(userId);
  res.on('finish', () => requestLocks.delete(userId));
  next();
};

const app = express();
app.set('trust proxy', 1); // Trust the first proxy to fix express-rate-limit behind a reverse proxy (like Render)
app.use(cors());
app.use(express.json());

// ── KNOWLEDGE BASE LOAD ───────────────────────────────
let knowledgeBase = { pages: [] };
try {
  const kbPath = path.join(__dirname, '../knowledge_base_v2.json');
  const kbData = fs.readFileSync(kbPath, 'utf8');
  knowledgeBase = JSON.parse(kbData);
  console.log(`Loaded knowledge base with ${knowledgeBase.pages.length} pages.`);
} catch (err) {
  console.error('Failed to load knowledge base:', err.message);
}

// ── KNOWLEDGE BASE RETRIEVAL ──────────────────────────────
function getRelevantContent(query) {
  if (!query || !knowledgeBase || !knowledgeBase.pages || knowledgeBase.pages.length === 0) {
    return "No specific knowledge base content available.";
  }

  const queryLower = query.toLowerCase();

  // Find pages where keywords, tags, or category match the query
  const matchedPages = knowledgeBase.pages.filter(page => {
    // Check keywords
    if (page.keywords && page.keywords.some(k => queryLower.includes(k.toLowerCase()))) return true;
    // Check tags
    if (page.tags && page.tags.some(t => queryLower.includes(t.toLowerCase()))) return true;
    // Check category
    if (page.category && queryLower.includes(page.category.toLowerCase())) return true;

    // Check section titles as fallback
    if (page.sections && page.sections.some(s => s.section_title && queryLower.includes(s.section_title.toLowerCase()))) return true;

    return false;
  });

  if (matchedPages.length === 0) {
    // Fallback to overview page if no direct match
    const fallbackPage = knowledgeBase.pages.find(p => p.category === 'overview') || knowledgeBase.pages[0];
    if (fallbackPage) {
      return `--- ${fallbackPage.title} ---\n${fallbackPage.full_text || ''}`;
    }
    return "No specific keyword match found in the knowledge base.";
  }

  // Combine content from matched pages
  const content = matchedPages.map(page => `--- ${page.title} ---\n${page.full_text || ''}`).join('\n\n');
  return content;
}

// ── SYSTEM PROMPT ────────────────────────────────────────
const BOT_NAME = "Avishek Ganguly";
const SYSTEM_PROMPT = `
You are ${BOT_NAME}, a friendly and professional AI assistant for VDart — a global staffing and technology company.
You represent VDart, VDart Digital, VDart Academy, and Sidd Ahmed (CEO of VDart).
You will be given content scraped from their official web pages.

═══════════════════════════════════════════
KNOWLEDGE AND ANSWERING RULES
═══════════════════════════════════════════
1. Always try to answer using the provided website content first.
2. If the answer is not in the website content, YOU MUST USE the Google Search tool to find information directly related to VDart, VDart Digital, VDart Academy, or Sidd Ahmed (e.g., who is the CEO).
3. Do NOT use Google Search for sensitive topics or topics unrelated to VDart. If the question is unrelated, refuse politely.
4. Never make up information. If you are unsure and the search yields no relevant results, say so and direct the user to contact the team.

═══════════════════════════════════════════
TOPICS YOU MUST REFUSE TO ANSWER
═══════════════════════════════════════════
- General knowledge, math, coding help, definitions, or trivia unrelated to VDart
- Questions about competitors or other companies
- Legal advice, salary negotiations, or compensation details
- Questions about layoffs, internal company issues, or confidential matters
- Medical, financial, or personal advice
- Anything that is not directly related to VDart and its services

For ALL refused topics, reply with exactly:
"I don't have that information available. For further assistance please contact us at csm@vdartinc.com or call (470) 323-8433 and our team will be happy to help."

═══════════════════════════════════════════
TONE AND PERSONALITY
═══════════════════════════════════════════
- Be warm, professional, and approachable at all times
- ABSOLUTELY DO NOT start your response with "Hi", "Hello", or any greeting. Jump straight to the answer.
- Use clear, plain language — avoid corporate jargon
- Keep all responses under 150 words unless a detailed list is specifically needed
- Never use markdown bold (asterisks like **text**) or bullet point asterisks (*) in responses
- Write in clean plain text only
- If a user writes in another language, respond in that same language

═══════════════════════════════════════════
HANDLING SPECIFIC SITUATIONS
═══════════════════════════════════════════
JOB APPLICATIONS:
- Provide details about the role or team if available
- Direct candidates to: https://vdart.com/careers
- Never promise a job or interview

COMPLAINTS OR FRUSTRATION:
- Acknowledge the frustration calmly: "I understand your frustration and I am sorry to hear that."
- Always escalate to: csm@vdartinc.com or (470) 323-8433
- Never argue or become defensive

RUDE OR ABUSIVE MESSAGES:
- Respond once calmly: "I am here to help with VDart-related questions. Please keep the conversation respectful."
- If it continues, say: "I am unable to continue this conversation. Please contact us directly at csm@vdartinc.com."

GREETINGS AND INTRODUCTIONS:
- NEVER introduce yourself or say "Hi/Hello" UNLESS the user's message is ONLY a greeting (like "Hi" or "Hello") or explicitly asks "Who are you?".
- If the user asks a question, skip the greeting completely and just answer the question directly.

FOLLOW-UP AND CLARIFICATION:
- If a question is vague, ask one short clarifying question before answering
- Example: "Are you asking about VDart staffing services or VDart Digital technology services?"

SENSITIVE TOPICS (salaries, layoffs, legal, internal matters):
- Respond: "That is something our team can better assist you with. Please reach out at csm@vdartinc.com or call (470) 323-8433."

═══════════════════════════════════════════
CONTACT DETAILS (always use these exactly)
═══════════════════════════════════════════
Email: csm@vdartinc.com
Phone: (470) 323-8433
Careers: https://vdart.com/careers
`;

// ── GEMINI API INTEGRATION ───────────────────────────────
async function callGeminiAPI(messages) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not set. Please configure it in the .env file.");
  }
  if (!apiKey.startsWith("AIzaSy") && !apiKey.startsWith("AQ.")) {
    throw new Error("Invalid GEMINI_API_KEY format. Gemini API keys must start with 'AIzaSy' or 'AQ.'. Please get a valid API key from https://aistudio.google.com/app/apikey.");
  }

  // Format messages into Gemini API format: { role, parts: [{ text }] }
  const formattedContents = messages.map((msg) => {
    const role = (msg.role === "assistant" || msg.role === "model") ? "model" : "user";
    const text = msg.content || msg.text || "";
    return {
      role,
      parts: [{ text }]
    };
  });

  // Using gemini-1.5-flash as the default standard model
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      contents: formattedContents,
      systemInstruction: {
        parts: [{ text: SYSTEM_PROMPT }]
      },
      tools: [{ googleSearch: {} }]
    })
  });

  if (!response.ok) {
    const errText = await response.text();
    // Provide a clearer error message for the ACCESS_TOKEN_TYPE_UNSUPPORTED error on AQ keys
    if (errText.includes("ACCESS_TOKEN_TYPE_UNSUPPORTED") || errText.includes("API_KEY_SERVICE_BLOCKED")) {
      throw new Error(`Gemini API Authentication Failed: Your API key is restricted or blocked for this service. Please generate a new key from https://aistudio.google.com/app/apikey`);
    }
    throw new Error(`Gemini API returned ${response.status}: ${errText}`);
  }

  const data = await response.json();
  if (
    data.candidates &&
    data.candidates[0] &&
    data.candidates[0].content &&
    data.candidates[0].content.parts &&
    data.candidates[0].content.parts[0]
  ) {
    return data.candidates[0].content.parts[0].text;
  } else {
    throw new Error("Invalid response format received from Gemini API");
  }
}

// ── CHAT ENDPOINT ────────────────────────────────────────
app.post("/chat", apiLimiter, preventParallelRequests, async (req, res) => {
  try {
    const { message, history = [] } = req.body;

    if (!message) {
      return res.status(400).json({ error: "No message provided" });
    }

    if (knowledgeBase.pages.length === 0) {
      return res.json({ reply: "Hi! I am currently unable to access my knowledge base. Please try again later." });
    }

    // ── CACHE LOOKUP ──
    const cached = await getCachedResponse(message);
    if (cached) {
      return res.json({ reply: cached });
    }

    const websiteContent = getRelevantContent(message);

    // ── TOKEN OPTIMIZATION / MEMORY TRIMMING ──
    // Keep only the last 4 messages in history to save tokens
    const trimmedHistory = history.slice(-4);

    // Optimize website content
    const optimizedContent = websiteContent.length > 15000 ? websiteContent.substring(0, 15000) + '... (truncated)' : websiteContent;

    const messages = [
      ...trimmedHistory,
      {
        role: "user",
        content: `Website content:\n${optimizedContent}\n\nUser question: ${message}`
      }
    ];

    // ── GENERATE RESPONSE ──
    const reply = await generateChatResponse({
      messages,
      systemInstruction: SYSTEM_PROMPT
    });
    const cleanedReply = (reply || "").replace(/\*/g, "");

    // ── CACHE THE RESPONSE ──
    await setCachedResponse(message, cleanedReply);

    res.json({ reply: cleanedReply });

  } catch (err) {
    console.error("Error:", err.message);
    res.status(500).json({ error: "Something went wrong" });
  }
});

// ── TEST ROUTE ───────────────────────────────────────────
app.get("/", (req, res) => {
  res.json({ status: "Server is running" });
});

// ── GET /testing-ai (Serve Premium HTML Test UI) ───────
app.get("/testing-ai", (req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gemini API Test Panel</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-color: #0b0f19;
      --card-bg: rgba(17, 24, 39, 0.75);
      --border-color: rgba(255, 255, 255, 0.08);
      --accent-color: #6366f1;
      --accent-hover: #4f46e5;
      --accent-glow: rgba(99, 102, 241, 0.4);
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --success-color: #10b981;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    body {
      background-color: var(--bg-color);
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 2rem;
      overflow-x: hidden;
      position: relative;
    }

    /* Ambient background glows */
    body::before {
      content: '';
      position: absolute;
      width: 500px;
      height: 500px;
      background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, rgba(0,0,0,0) 70%);
      top: -10%;
      left: -10%;
      z-index: 0;
      pointer-events: none;
    }

    body::after {
      content: '';
      position: absolute;
      width: 600px;
      height: 600px;
      background: radial-gradient(circle, rgba(16, 185, 129, 0.06) 0%, rgba(0,0,0,0) 70%);
      bottom: -10%;
      right: -10%;
      z-index: 0;
      pointer-events: none;
    }

    .container {
      width: 100%;
      max-width: 800px;
      z-index: 10;
    }

    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border-color);
      border-radius: 24px;
      padding: 3rem;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 
                  0 0 0 1px rgba(255, 255, 255, 0.02) inset;
      position: relative;
      overflow: hidden;
    }

    .card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 4px;
      background: linear-gradient(90deg, #6366f1, #10b981, #a855f7);
    }

    header {
      margin-bottom: 2.5rem;
      text-align: center;
    }

    h1 {
      font-size: 2.25rem;
      font-weight: 700;
      background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
      letter-spacing: -0.025em;
    }

    .subtitle {
      color: var(--text-muted);
      font-size: 1rem;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.2);
      color: var(--success-color);
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 1.5rem;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      background-color: var(--success-color);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--success-color);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0% { transform: scale(0.95); opacity: 0.5; }
      55% { transform: scale(1.1); opacity: 1; }
      100% { transform: scale(0.95); opacity: 0.5; }
    }

    .form-group {
      margin-bottom: 1.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    label {
      font-size: 0.85rem;
      font-weight: 600;
      color: #cbd5e1;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }

    .input-wrapper {
      position: relative;
    }

    input[type="text"], select {
      width: 100%;
      background: rgba(10, 15, 26, 0.8);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 1rem 1.25rem;
      color: var(--text-main);
      font-family: inherit;
      font-size: 1rem;
      outline: none;
    }

    input[type="text"]:focus, select:focus {
      border-color: var(--accent-color);
      box-shadow: 0 0 15px var(--accent-glow);
    }

    .btn {
      width: 100%;
      background: linear-gradient(135deg, var(--accent-color) 0%, var(--accent-hover) 100%);
      color: #ffffff;
      border: none;
      border-radius: 12px;
      padding: 1.1rem;
      font-size: 1.05rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 0.75rem;
      box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
      position: relative;
      overflow: hidden;
    }

    .btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5);
    }

    .btn:active {
      transform: translateY(0);
    }

    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    .output-container {
      margin-top: 2.5rem;
      display: none;
      flex-direction: column;
      gap: 0.75rem;
      animation: fadeIn 0.4s ease-out forwards;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .output-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .output-box {
      width: 100%;
      background: rgba(10, 15, 26, 0.95);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 1.5rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.95rem;
      line-height: 1.6;
      white-space: pre-wrap;
      color: #e2e8f0;
      min-height: 120px;
      max-height: 400px;
      overflow-y: auto;
      box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
    }

    .spinner {
      width: 20px;
      height: 20px;
      border: 3px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: #ffffff;
      animation: spin 1s ease-in-out infinite;
      display: none;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .meta-info {
      font-size: 0.8rem;
      color: var(--text-muted);
      display: flex;
      gap: 1.5rem;
    }

    .meta-item {
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }
  </style>
</head>
<body>

  <div class="container">
    <div class="card">
      <header>
        <div class="status-badge">
          <div class="status-dot"></div>
          Gemini SDK Connected
        </div>
        <h1>Gemini API Live Tester</h1>
        <p class="subtitle">Direct interface to check model outputs using the new @google/genai SDK</p>
      </header>

      <div class="form-group">
        <label for="prompt">Test Prompt</label>
        <div class="input-wrapper">
          <input type="text" id="prompt" value="Explain how AI works in a few words" placeholder="Enter your prompt here...">
        </div>
      </div>

      <div class="form-group">
        <label for="model">Model</label>
        <select id="model">
          <option value="gemini-2.5-flash">gemini-2.5-flash (Recommended)</option>
          <option value="gemini-1.5-flash">gemini-1.5-flash</option>
          <option value="gemini-2.5-pro">gemini-2.5-pro</option>
        </select>
      </div>

      <button id="runBtn" class="btn" onclick="runTest()">
        <span class="spinner" id="spinner"></span>
        <span id="btnText">Generate Content</span>
      </button>

      <div class="output-container" id="outputContainer">
        <div class="output-header">
          <label>API Response</label>
          <div class="meta-info">
            <div class="meta-item">⏱️ Latency: <span id="latency">0ms</span></div>
            <div class="meta-item">🤖 Model: <span id="resModel">N/A</span></div>
          </div>
        </div>
        <div class="output-box" id="outputBox"></div>
      </div>
    </div>
  </div>

  <script>
    async function runTest() {
      const prompt = document.getElementById('prompt').value.trim();
      const model = document.getElementById('model').value;
      
      const runBtn = document.getElementById('runBtn');
      const spinner = document.getElementById('spinner');
      const btnText = document.getElementById('btnText');
      const outputContainer = document.getElementById('outputContainer');
      const outputBox = document.getElementById('outputBox');
      const latencySpan = document.getElementById('latency');
      const resModelSpan = document.getElementById('resModel');

      if (!prompt) {
        alert("Please enter a prompt!");
        return;
      }

      // UI loading state
      runBtn.disabled = true;
      spinner.style.display = 'block';
      btnText.textContent = 'Generating...';
      outputContainer.style.display = 'none';

      const startTime = Date.now();

      try {
        const response = await fetch('/testing-ai', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ prompt, model })
        });

        const data = await response.json();
        const endTime = Date.now();
        const latency = endTime - startTime;

        outputContainer.style.display = 'flex';
        latencySpan.textContent = latency + 'ms';
        resModelSpan.textContent = model;

        if (response.ok) {
          outputBox.textContent = data.text;
          outputBox.style.color = '#e2e8f0';
        } else {
          outputBox.textContent = "Error: " + (data.error || "Failed to fetch response");
          outputBox.style.color = '#ef4444';
        }

      } catch (err) {
        outputContainer.style.display = 'flex';
        outputBox.textContent = "Error: " + err.message;
        outputBox.style.color = '#ef4444';
        latencySpan.textContent = 'N/A';
        resModelSpan.textContent = model;
      } finally {
        runBtn.disabled = false;
        spinner.style.display = 'none';
        btnText.textContent = 'Generate Content';
      }
    }
  </script>
</body>
</html>`);
});

// ── POST /testing-ai (Execute Gemini SDK Call with VDart System Rules) ────────────
app.post("/testing-ai", apiLimiter, preventParallelRequests, async (req, res) => {
  try {
    const { prompt = "Explain how AI works in a few words", model = "gemini-2.5-flash" } = req.body;

    if (knowledgeBase.pages.length === 0) {
      return res.json({ text: "Hi! I am currently unable to access my knowledge base. Please try again later." });
    }

    // ── CACHE LOOKUP ──
    const cached = await getCachedResponse(prompt);
    if (cached) {
      return res.json({ text: cached });
    }

    // Fetch the relevant knowledge base content
    const websiteContent = getRelevantContent(prompt);
    const optimizedContent = websiteContent.length > 15000 ? websiteContent.substring(0, 15000) + '... (truncated)' : websiteContent;

    // ── GENERATE RESPONSE ──
    const reply = await generateChatResponse({
      prompt: `Website content:\n${optimizedContent}\n\nUser question: ${prompt}`,
      model,
      systemInstruction: SYSTEM_PROMPT
    });
    const cleanedText = (reply || "").replace(/\*/g, "");

    // ── CACHE THE RESPONSE ──
    await setCachedResponse(prompt, cleanedText);

    res.json({ text: cleanedText });
  } catch (err) {
    console.error("SDK Test Error:", err.message);

    // Provide a clearer error message for the ACCESS_TOKEN_TYPE_UNSUPPORTED error on AQ keys
    let errorMsg = err.message;
    if (errorMsg.includes("ACCESS_TOKEN_TYPE_UNSUPPORTED")) {
      errorMsg = "Authentication Failed: Your API key is restricted or blocked for this service. Please generate a new key from https://aistudio.google.com/app/apikey";
    }

    res.status(500).json({ error: errorMsg });
  }
});

// ── FIREBASE DATA VIEWER ENDPOINT ────────────────────────
const FIREBASE_DB_URL = "https://vdartvallet-default-rtdb.asia-southeast1.firebasedatabase.app/.json";

app.get("/backend", (req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VDart Vallet — Firebase Data Explorer</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #06090f;
      --surface: rgba(14, 20, 33, 0.85);
      --surface-raised: rgba(22, 30, 48, 0.9);
      --border: rgba(255,255,255,0.06);
      --border-hover: rgba(255,255,255,0.12);
      --accent: #7c3aed;
      --accent-soft: rgba(124, 58, 237, 0.15);
      --accent2: #06b6d4;
      --accent2-soft: rgba(6, 182, 212, 0.12);
      --accent3: #f59e0b;
      --accent3-soft: rgba(245, 158, 11, 0.12);
      --green: #10b981;
      --green-soft: rgba(16, 185, 129, 0.12);
      --red: #ef4444;
      --red-soft: rgba(239, 68, 68, 0.1);
      --text: #f1f5f9;
      --text-dim: #94a3b8;
      --text-muted: #64748b;
      --radius: 16px;
      --radius-sm: 10px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* ── Ambient blurs ── */
    body::before, body::after {
      content: '';
      position: fixed;
      border-radius: 50%;
      pointer-events: none;
      filter: blur(100px);
      z-index: 0;
    }
    body::before {
      width: 600px; height: 600px;
      background: rgba(124,58,237,0.08);
      top: -200px; left: -100px;
    }
    body::after {
      width: 500px; height: 500px;
      background: rgba(6,182,212,0.06);
      bottom: -150px; right: -100px;
    }

    .app { position: relative; z-index: 1; max-width: 1280px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }

    /* ── Header ── */
    .header {
      text-align: center;
      margin-bottom: 2.5rem;
      padding-bottom: 2rem;
      border-bottom: 1px solid var(--border);
    }
    .header-badge {
      display: inline-flex; align-items: center; gap: 8px;
      background: var(--green-soft);
      border: 1px solid rgba(16,185,129,0.2);
      color: var(--green);
      font-size: 0.8rem; font-weight: 600;
      padding: 6px 14px; border-radius: 999px;
      margin-bottom: 1rem;
    }
    .header-badge .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 8px var(--green);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%,100% { opacity:.5; transform:scale(.9) }
      50% { opacity:1; transform:scale(1.15) }
    }
    .header h1 {
      font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em;
      background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .header p { color: var(--text-dim); margin-top: 0.4rem; font-size: 0.95rem; }

    /* ── Stats bar ── */
    .stats-bar {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }
    .stat-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem 1.5rem;
      backdrop-filter: blur(12px);
    }
    .stat-card .stat-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
    .stat-card .stat-value { font-size: 1.8rem; font-weight: 800; margin-top: 4px; }
    .stat-card.purple .stat-value { color: var(--accent); }
    .stat-card.cyan .stat-value { color: var(--accent2); }
    .stat-card.amber .stat-value { color: var(--accent3); }
    .stat-card.green .stat-value { color: var(--green); }

    /* ── Toolbar ── */
    .toolbar {
      display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap; align-items: center;
    }
    .search-box {
      flex: 1; min-width: 220px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 0.75rem 1rem;
      color: var(--text);
      font-family: inherit; font-size: 0.9rem;
      outline: none;
    }
    .search-box:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    .search-box::placeholder { color: var(--text-muted); }
    .btn-toolbar {
      padding: 0.75rem 1.25rem;
      background: var(--surface-raised);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      color: var(--text-dim);
      font-family: inherit; font-size: 0.85rem; font-weight: 600;
      cursor: pointer;
    }
    .btn-toolbar:hover { border-color: var(--border-hover); color: var(--text); background: rgba(30,40,60,0.9); }
    .btn-toolbar.active { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
    .btn-refresh {
      padding: 0.75rem 1.5rem;
      background: linear-gradient(135deg, var(--accent), #6d28d9);
      border: none; border-radius: var(--radius-sm);
      color: #fff; font-family: inherit; font-size: 0.85rem; font-weight: 700;
      cursor: pointer;
      box-shadow: 0 4px 16px rgba(124,58,237,0.3);
    }
    .btn-refresh:hover { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(124,58,237,0.45); }

    /* ── Collection cards grid ── */
    .collections-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }
    .collection-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem 1.5rem;
      cursor: pointer;
      backdrop-filter: blur(12px);
      transition: all 0.2s ease;
    }
    .collection-card:hover {
      border-color: var(--border-hover);
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .collection-card.expanded {
      grid-column: 1 / -1;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }
    .collection-header {
      display: flex; justify-content: space-between; align-items: center;
    }
    .collection-name {
      font-size: 1.05rem; font-weight: 700;
      display: flex; align-items: center; gap: 8px;
    }
    .collection-icon {
      width: 32px; height: 32px; border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      font-size: 0.9rem;
    }
    .collection-meta {
      display: flex; gap: 1rem; margin-top: 0.75rem;
    }
    .meta-tag {
      font-size: 0.75rem; color: var(--text-muted); font-weight: 500;
      background: rgba(255,255,255,0.04);
      padding: 3px 10px; border-radius: 6px;
    }
    .collection-chevron {
      font-size: 1.2rem; color: var(--text-muted);
      transition: transform 0.25s ease;
    }
    .collection-card.expanded .collection-chevron { transform: rotate(180deg); }

    /* ── Data viewer (tree) ── */
    .data-viewer {
      margin-top: 1rem;
      max-height: 600px;
      overflow-y: auto;
      display: none;
    }
    .collection-card.expanded .data-viewer { display: block; }

    .data-viewer::-webkit-scrollbar { width: 6px; }
    .data-viewer::-webkit-scrollbar-track { background: transparent; }
    .data-viewer::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

    .tree-node { padding-left: 1.25rem; border-left: 1px solid var(--border); margin-left: 0.5rem; }
    .tree-row {
      display: flex; align-items: flex-start; gap: 6px;
      padding: 4px 0; font-size: 0.85rem;
      line-height: 1.5;
    }
    .tree-toggle {
      cursor: pointer; user-select: none;
      color: var(--text-muted); font-size: 0.7rem;
      width: 16px; flex-shrink: 0;
      margin-top: 3px;
    }
    .tree-key {
      color: var(--accent2);
      font-family: 'JetBrains Mono', monospace;
      font-weight: 500;
      flex-shrink: 0;
    }
    .tree-sep { color: var(--text-muted); flex-shrink: 0; }
    .tree-value {
      font-family: 'JetBrains Mono', monospace;
      word-break: break-all;
    }
    .tree-value.str { color: #a5d6a7; }
    .tree-value.num { color: #ffcc80; }
    .tree-value.bool { color: #ce93d8; }
    .tree-value.null { color: var(--text-muted); font-style: italic; }
    .tree-value.type-badge {
      font-size: 0.7rem; color: var(--text-muted);
      background: rgba(255,255,255,0.04);
      padding: 1px 8px; border-radius: 4px;
      font-family: inherit;
    }

    /* ── Loading & Error ── */
    .loading-overlay {
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      padding: 4rem 2rem; text-align: center;
    }
    .loader {
      width: 48px; height: 48px;
      border: 4px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin-bottom: 1.5rem;
    }
    @keyframes spin { to { transform: rotate(360deg) } }
    .loading-text { color: var(--text-dim); font-size: 0.95rem; }

    .error-box {
      background: var(--red-soft);
      border: 1px solid rgba(239,68,68,0.2);
      border-radius: var(--radius);
      padding: 1.5rem;
      color: #fca5a5;
      text-align: center;
    }

    /* ── Responsive ── */
    @media (max-width: 640px) {
      .header h1 { font-size: 1.6rem; }
      .stats-bar { grid-template-columns: 1fr 1fr; }
      .collections-grid { grid-template-columns: 1fr; }
      .app { padding: 1rem; }
    }
  </style>
</head>
<body>
<div class="app">
  <div class="header">
    <div class="header-badge"><div class="dot"></div> Firebase Realtime Database</div>
    
  </div>

  <div class="stats-bar" id="statsBar"></div>

  <div class="toolbar">
    <input type="text" class="search-box" id="searchBox" placeholder="Search keys or values..." oninput="filterData()">
    <button class="btn-toolbar" id="btnExpandAll" onclick="toggleExpandAll()">Expand All</button>
    <button class="btn-refresh" onclick="loadData()">↻ Refresh</button>
  </div>

  <div id="mainContent">
    <div class="loading-overlay" id="loader">
      <div class="loader"></div>
      <div class="loading-text">Fetching data from Firebase...</div>
    </div>
  </div>

  <div class="collections-grid" id="collectionsGrid" style="display:none;"></div>
</div>

<script>
  const FIREBASE_URL = "${FIREBASE_DB_URL}";
  let fullData = null;
  let allExpanded = false;

  // ── Icon & color mapping for known collections ──
  const collectionMeta = {
    config: { icon: '⚙️', color: 'var(--accent)' },
    users: { icon: '👤', color: 'var(--accent2)' },
    creditTransactions: { icon: '💳', color: 'var(--green)' },
    debitTransactions: { icon: '📤', color: 'var(--accent3)' },
    stores: { icon: '🏪', color: '#ec4899' },
    foodMenu: { icon: '🍔', color: '#f97316' },
    storeOrders: { icon: '📦', color: '#8b5cf6' },
    attendance: { icon: '📋', color: '#14b8a6' },
    userAttendance: { icon: '🕐', color: '#06b6d4' },
    feedback: { icon: '💬', color: '#a855f7' },
  };

  function getMeta(key) {
    return collectionMeta[key] || { icon: '📁', color: 'var(--text-dim)' };
  }

  // ── Count items ──
  function countItems(val) {
    if (val === null || val === undefined) return 0;
    if (typeof val === 'object') return Object.keys(val).length;
    return 1;
  }

  function detectType(val) {
    if (val === null || val === undefined) return 'null';
    if (Array.isArray(val)) return 'array';
    return typeof val;
  }

  function totalDeepCount(val) {
    if (val === null || typeof val !== 'object') return 1;
    let c = 0;
    for (const k of Object.keys(val)) {
      c += totalDeepCount(val[k]);
    }
    return c;
  }

  // ── Build tree HTML ──
  function buildTree(data, depth = 0, maxInitialDepth = 1) {
    if (data === null || data === undefined) return '<span class="tree-value null">null</span>';
    if (typeof data !== 'object') {
      const t = typeof data;
      const cls = t === 'string' ? 'str' : t === 'number' ? 'num' : t === 'boolean' ? 'bool' : '';
      const display = t === 'string' ? '"' + escapeHtml(String(data)) + '"' : String(data);
      return '<span class="tree-value ' + cls + '">' + display + '</span>';
    }

    const keys = Object.keys(data);
    if (keys.length === 0) return '<span class="tree-value null">{}</span>';

    const collapsed = depth >= maxInitialDepth;
    let html = '';

    keys.forEach(key => {
      const val = data[key];
      const isObj = val !== null && typeof val === 'object';
      const childCount = isObj ? Object.keys(val).length : 0;

      html += '<div class="tree-row">';
      if (isObj && childCount > 0) {
        html += '<span class="tree-toggle" onclick="toggleNode(this)">' + (collapsed ? '▶' : '▼') + '</span>';
      } else {
        html += '<span class="tree-toggle"></span>';
      }
      html += '<span class="tree-key">' + escapeHtml(key) + '</span>';
      html += '<span class="tree-sep">: </span>';

      if (isObj && childCount > 0) {
        html += '<span class="tree-value type-badge">' + (Array.isArray(val) ? 'Array' : 'Object') + ' (' + childCount + ')</span>';
        html += '</div>';
        html += '<div class="tree-node" style="display:' + (collapsed ? 'none' : 'block') + ';">';
        html += buildTree(val, depth + 1, maxInitialDepth);
        html += '</div>';
      } else {
        html += buildTree(val, depth + 1, maxInitialDepth);
        html += '</div>';
      }
    });

    return html;
  }

  function escapeHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  function toggleNode(el) {
    const sibling = el.closest('.tree-row').nextElementSibling;
    if (!sibling || !sibling.classList.contains('tree-node')) return;
    const hidden = sibling.style.display === 'none';
    sibling.style.display = hidden ? 'block' : 'none';
    el.textContent = hidden ? '▼' : '▶';
  }

  function toggleExpandAll() {
    allExpanded = !allExpanded;
    const btn = document.getElementById('btnExpandAll');
    btn.textContent = allExpanded ? 'Collapse All' : 'Expand All';
    btn.classList.toggle('active', allExpanded);
    document.querySelectorAll('.tree-node').forEach(n => n.style.display = allExpanded ? 'block' : 'none');
    document.querySelectorAll('.tree-toggle').forEach(t => { if(t.textContent) t.textContent = allExpanded ? '▼' : '▶'; });
  }

  // ── Render stats ──
  function renderStats(data) {
    const bar = document.getElementById('statsBar');
    const keys = Object.keys(data);
    const totalRecords = keys.reduce((sum, k) => sum + countItems(data[k]), 0);
    const totalFields = totalDeepCount(data);

    const colors = ['purple','cyan','amber','green'];
    const stats = [
      { label: 'Collections', value: keys.length },
      { label: 'Top-Level Records', value: totalRecords.toLocaleString() },
      { label: 'Total Fields', value: totalFields.toLocaleString() },
      { label: 'Status', value: 'Live' },
    ];

    bar.innerHTML = stats.map((s,i) =>
      '<div class="stat-card ' + colors[i % 4] + '">' +
        '<div class="stat-label">' + s.label + '</div>' +
        '<div class="stat-value">' + s.value + '</div>' +
      '</div>'
    ).join('');
  }

  // ── Render collection cards ──
  function renderCollections(data) {
    const grid = document.getElementById('collectionsGrid');
    const keys = Object.keys(data);

    grid.innerHTML = keys.map(key => {
      const meta = getMeta(key);
      const val = data[key];
      const count = countItems(val);
      const type = detectType(val);

      return '<div class="collection-card" data-key="' + key + '" onclick="toggleCollection(this, \\'' + key + '\\')">' +
        '<div class="collection-header">' +
          '<div class="collection-name">' +
            '<div class="collection-icon" style="background:' + meta.color + '22;">' + meta.icon + '</div>' +
            escapeHtml(key) +
          '</div>' +
          '<span class="collection-chevron">▼</span>' +
        '</div>' +
        '<div class="collection-meta">' +
          '<span class="meta-tag">' + count + ' items</span>' +
          '<span class="meta-tag">Type: ' + type + '</span>' +
        '</div>' +
        '<div class="data-viewer" id="viewer-' + key + '"></div>' +
      '</div>';
    }).join('');

    grid.style.display = 'grid';
  }

  function toggleCollection(card, key) {
    const wasExpanded = card.classList.contains('expanded');

    // Collapse all first if not searching
    const q = document.getElementById('searchBox').value.trim();
    if (!q) {
      document.querySelectorAll('.collection-card.expanded').forEach(c => c.classList.remove('expanded'));
    }

    if (!wasExpanded) {
      card.classList.add('expanded');
      const viewer = document.getElementById('viewer-' + key);
      if (!viewer.innerHTML) {
        viewer.innerHTML = '<div style="padding:0.5rem 0;">' + buildTree(filteredData[key], 0, 1) + '</div>';
      }
    } else if (!q) {
      card.classList.remove('expanded');
    }
  }

  // ── Search / Filter ──
  function filterObject(obj, q) {
    if (obj === null || obj === undefined) return null;
    if (typeof obj !== 'object') {
      return String(obj).toLowerCase().includes(q) ? obj : null;
    }
    const isArray = Array.isArray(obj);
    let result = isArray ? [] : {};
    let hasMatch = false;
    for (const key of Object.keys(obj)) {
      if (key.toLowerCase().includes(q)) {
        result[key] = obj[key];
        hasMatch = true;
      } else {
        const filteredChild = filterObject(obj[key], q);
        if (filteredChild !== null) {
          result[key] = filteredChild;
          hasMatch = true;
        }
      }
    }
    return hasMatch ? result : null;
  }

  function filterData() {
    const q = document.getElementById('searchBox').value.toLowerCase().trim();
    if (!q) {
      filteredData = fullData;
      renderCollections(filteredData);
      return;
    }
    
    const newFiltered = {};
    for (const key of Object.keys(fullData)) {
      if (key.toLowerCase().includes(q)) {
        newFiltered[key] = fullData[key];
      } else {
        const filteredChild = filterObject(fullData[key], q);
        if (filteredChild !== null) {
          newFiltered[key] = filteredChild;
        }
      }
    }
    filteredData = newFiltered;
    renderCollections(filteredData);
    
    // Auto-expand all and render deep trees
    document.querySelectorAll('.collection-card').forEach(card => {
       const k = card.getAttribute('data-key');
       card.classList.add('expanded');
       const viewer = document.getElementById('viewer-' + k);
       viewer.innerHTML = '<div style="padding:0.5rem 0;">' + buildTree(filteredData[k], 0, 999) + '</div>';
    });
  }

  // ── Load data ──
  async function loadData() {
    const content = document.getElementById('mainContent');
    const grid = document.getElementById('collectionsGrid');

    content.innerHTML = '<div class="loading-overlay"><div class="loader"></div><div class="loading-text">Fetching data from Firebase...</div></div>';
    grid.style.display = 'none';

    try {
      const res = await fetch(FIREBASE_URL);
      if (!res.ok) throw new Error('HTTP ' + res.status);
      fullData = await res.json();
      filteredData = fullData;

      content.innerHTML = '';
      renderStats(fullData);
      renderCollections(fullData);
      
      // Re-apply filter if search box has text
      if (document.getElementById('searchBox').value.trim()) {
        filterData();
      }
    } catch (err) {
      content.innerHTML = '<div class="error-box">Failed to fetch Firebase data: ' + escapeHtml(err.message) + '</div>';
    }
  }

  // ── Init ──
  loadData();
</script>
</body>
</html>`);
});

// ── FIREBASE DATA API ENDPOINT (JSON) ────────────────────
app.get("/backend/api", async (req, res) => {
  try {
    const response = await fetch(FIREBASE_DB_URL);
    if (!response.ok) {
      const errText = await response.text();
      return res.status(response.status).json({ error: "Firebase fetch failed: " + errText });
    }
    const data = await response.json();

    // Build summary
    const summary = {};
    for (const [key, value] of Object.entries(data)) {
      const type = Array.isArray(value) ? 'array' : typeof value;
      const count = (value && typeof value === 'object') ? Object.keys(value).length : 1;
      summary[key] = { type, count };
    }

    res.json({ summary, data });
  } catch (err) {
    console.error("Firebase API Error:", err.message);
    res.status(500).json({ error: "Failed to fetch Firebase data: " + err.message });
  }
});

// ── START SERVER ─────────────────────────────────────────
if (process.env.NODE_ENV !== 'production') {
  const PORT = process.env.PORT || 3000;
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
}
module.exports = app;