import express from "express";
import cors from "cors";

const app = express();
const PORT = 4000;

app.use(cors());
app.use(express.json());

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

app.post("/api/applications", async (req, res) => {
  const mode = req.query.mode || "success";
  const baseDelay = 700;

  if (mode === "timeout") {
    await delay(10000);
    return res.status(504).json({ error: "TIMEOUT" });
  }

  await delay(baseDelay);

  if (mode === "error") {
    return res.status(500).json({ error: "SERVER_ERROR" });
  }

  if (mode === "random" && Math.random() < 0.3) {
    return res.status(503).json({ error: "UNAVAILABLE" });
  }

  return res.status(201).json({
    confirmationId: crypto.randomUUID(),
    received: req.body
  });
});

app.listen(PORT, () => {
  console.log(`Mock API listening on http://localhost:${PORT}`);
});
