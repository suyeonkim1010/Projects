import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();

const PORT = process.env.PORT || 5177;
const OLLAMA_HOST = process.env.OLLAMA_HOST || "http://127.0.0.1:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "llama3.1:8b";

app.use(express.json({ limit: "1mb" }));

const staticDir = path.resolve(__dirname, "..");
app.use(express.static(staticDir));

app.post("/summarize", async (req, res) => {
  const { jdText } = req.body || {};
  if (!jdText || typeof jdText !== "string") {
    return res.status(400).json({ error: "jdText is required" });
  }

  try {
    const response = await fetch(`${OLLAMA_HOST}/api/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        stream: false,
        format: "json",
        prompt: [
          "Extract key job info from the JD.",
          "Respond ONLY with minified JSON with keys:",
          "company, location, role, workMode (Remote/Hybrid/On-site/Unknown), compensation, companySummary, hiringFor, skills (array), tags (array).",
          "Keep summaries concise, 1-2 sentences. If unknown, use empty string or empty array.",
          "JD:",
          jdText,
        ].join("\n"),
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return res.status(500).json({ error: "Ollama request failed", details: errorText });
    }

    const data = await response.json();
    const message = data.response || "{}";

    let parsed;
    try {
      parsed = JSON.parse(message);
    } catch (error) {
      return res.status(500).json({ error: "Failed to parse model response", details: message });
    }

    return res.json({ data: parsed });
  } catch (error) {
    return res.status(500).json({ error: "Server error", details: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Job Tracker server running at http://localhost:${PORT}`);
});
