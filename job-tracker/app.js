const STORAGE_KEY = "jobTrackerEntriesV1";

const jdInput = document.getElementById("jdInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const duplicateWarning = document.getElementById("duplicateWarning");
const aiStatus = document.getElementById("aiStatus");
const syncStatus = document.getElementById("syncStatus");
const navAuthBtn = document.getElementById("navAuthBtn");

const companyInput = document.getElementById("companyInput");
const locationInput = document.getElementById("locationInput");
const roleInput = document.getElementById("roleInput");
const workModeInput = document.getElementById("workModeInput");
const compInput = document.getElementById("compInput");
const companySummaryInput = document.getElementById("companySummaryInput");
const hiringInput = document.getElementById("hiringInput");
const skillsInput = document.getElementById("skillsInput");
const tagsInput = document.getElementById("tagsInput");
const statusInput = document.getElementById("statusInput");
const deadlineInput = document.getElementById("deadlineInput");
const entryForm = document.getElementById("entryForm");
const resetBtn = document.getElementById("resetBtn");

const searchInput = document.getElementById("searchInput");
const statusFilter = document.getElementById("statusFilter");
const modeFilter = document.getElementById("modeFilter");
const exportBtn = document.getElementById("exportBtn");
const clearAllBtn = document.getElementById("clearAllBtn");
const entriesEl = document.getElementById("entries");
const statsEl = document.getElementById("stats");

let entries = loadEntries();
let editingId = null;
let isAnalyzing = false;
let isSyncing = false;
let currentUser = null;

const SUPABASE_URL = "https://ecqhywxtmlriuqctwtsd.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVjcWh5d3h0bWxyaXVxY3R3dHNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjcxNDY2MDUsImV4cCI6MjA4MjcyMjYwNX0.32CEEDovRUzLUDwY-EHx3dZIgEPqjVkerxgxhD1W3XE";
const SUPABASE_TABLE = "job_entries";
const supabaseClient = window.supabase?.createClient?.(SUPABASE_URL, SUPABASE_ANON_KEY) || null;

const skillKeywords = [
  "javascript",
  "typescript",
  "react",
  "vue",
  "angular",
  "node",
  "python",
  "java",
  "kotlin",
  "swift",
  "go",
  "rust",
  "c++",
  "c#",
  "sql",
  "aws",
  "gcp",
  "azure",
  "docker",
  "kubernetes",
  "figma",
  "sketch",
  "product",
  "ux",
  "ui",
  "ml",
  "ai",
  "data",
  "analytics",
  "ios",
  "android",
  "frontend",
  "backend",
  "fullstack",
  "devops",
  "security",
  "qa",
];

const companyHints = [
  /company\s*:\s*(.+)/i,
  /employer\s*:\s*(.+)/i,
  /회사\s*[:\-]\s*(.+)/i,
  /기업\s*[:\-]\s*(.+)/i,
];

const locationHints = [
  /location\s*:\s*(.+)/i,
  /based in\s+(.+)/i,
  /근무지\s*[:\-]\s*(.+)/i,
  /지역\s*[:\-]\s*(.+)/i,
];

const roleHints = [
  /title\s*:\s*(.+)/i,
  /role\s*:\s*(.+)/i,
  /position\s*:\s*(.+)/i,
  /채용\s*[:\-]\s*(.+)/i,
];

const compensationHints = [
  /(\$|USD|usd|연봉|salary).{0,40}(\d{2,3}[,\d]{0,6})/,
  /(\d{2,3}[,\d]{0,6})\s*(\$|usd|USD|만원|원)/,
];

const workModeHints = [
  { label: "Remote", re: /remote|재택/i },
  { label: "Hybrid", re: /hybrid|부분\s*재택/i },
  { label: "On-site", re: /on[-\s]?site|출근|사무실/i },
];

function loadEntries() {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (!stored) return [];
  try {
    return JSON.parse(stored);
  } catch (error) {
    console.warn("Failed to parse entries", error);
    return [];
  }
}

function saveEntries() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
}

function updateSyncStatus(message) {
  if (!syncStatus) return;
  syncStatus.textContent = message;
}

function hasSupabase() {
  return Boolean(supabaseClient);
}

function hasUser() {
  return Boolean(currentUser);
}

function updateAuthUI() {
  if (!navAuthBtn) return;
  if (hasUser()) {
    navAuthBtn.textContent = "Sign out";
    updateSyncStatus("Sync: ready");
  } else {
    navAuthBtn.textContent = "Sign in";
    updateSyncStatus(hasSupabase() ? "Sync: sign in to sync" : "Sync: local only");
  }
}

function toDbEntry(entry) {
  return {
    id: entry.id,
    user_id: currentUser?.id || null,
    created_at: entry.createdAt,
    jd_text: entry.jdText,
    company: entry.company,
    location: entry.location,
    role: entry.role,
    work_mode: entry.workMode,
    compensation: entry.compensation,
    company_summary: entry.companySummary,
    hiring_for: entry.hiringFor,
    skills: entry.skills,
    tags: entry.tags,
    status: entry.status,
    deadline: entry.deadline || null,
    fingerprint: entry.fingerprint,
  };
}

function fromDbEntry(row) {
  return {
    id: row.id,
    createdAt: row.created_at,
    jdText: row.jd_text || "",
    company: row.company || "",
    location: row.location || "",
    role: row.role || "",
    workMode: row.work_mode || "",
    compensation: row.compensation || "",
    companySummary: row.company_summary || "",
    hiringFor: row.hiring_for || "",
    skills: Array.isArray(row.skills) ? row.skills : [],
    tags: Array.isArray(row.tags) ? row.tags : [],
    status: row.status || "Not Applied",
    deadline: row.deadline || "",
    fingerprint: row.fingerprint || "",
  };
}

function mergeEntries(localEntries, remoteEntries) {
  const merged = new Map();
  for (const entry of remoteEntries) {
    merged.set(entry.id, entry);
  }
  for (const entry of localEntries) {
    merged.set(entry.id, entry);
  }
  return Array.from(merged.values());
}

async function fetchEntriesFromSupabase() {
  if (!hasSupabase() || !hasUser()) return [];
  const { data, error } = await supabaseClient
    .from(SUPABASE_TABLE)
    .select("*")
    .order("created_at", { ascending: false });
  if (error) throw error;
  return data.map(fromDbEntry);
}

async function upsertEntryToSupabase(entry) {
  if (!hasSupabase() || !hasUser()) return;
  const { error } = await supabaseClient
    .from(SUPABASE_TABLE)
    .upsert(toDbEntry(entry), { onConflict: "id" });
  if (error) throw error;
}

async function upsertEntriesToSupabase(entryList) {
  if (!hasSupabase() || !hasUser()) return;
  if (!entryList.length) return;
  const { error } = await supabaseClient
    .from(SUPABASE_TABLE)
    .upsert(entryList.map(toDbEntry), { onConflict: "id" });
  if (error) throw error;
}

async function deleteEntryFromSupabase(id) {
  if (!hasSupabase() || !hasUser()) return;
  const { error } = await supabaseClient
    .from(SUPABASE_TABLE)
    .delete()
    .eq("id", id);
  if (error) throw error;
}

async function deleteAllEntriesFromSupabase() {
  if (!hasSupabase() || !hasUser()) return;
  const { error } = await supabaseClient
    .from(SUPABASE_TABLE)
    .delete()
    .eq("user_id", currentUser.id);
  if (error) throw error;
}

async function runSync(task) {
  if (!hasSupabase() || !hasUser()) return;
  if (isSyncing) return;
  isSyncing = true;
  updateSyncStatus("Sync: updating...");
  try {
    await task();
    updateSyncStatus("Sync: up to date");
  } catch (error) {
    console.warn("Supabase sync failed", error);
    updateSyncStatus("Sync: error (local only)");
  } finally {
    isSyncing = false;
  }
}

function normalize(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9가-힣]+/g, " ")
    .trim();
}

function pickFromHints(text, hints) {
  for (const hint of hints) {
    const match = text.match(hint);
    if (match && match[1]) {
      return match[1].split("\n")[0].trim();
    }
  }
  return "";
}

function extractSection(text, headers) {
  const lines = text.split("\n");
  let capture = false;
  const section = [];

  for (const line of lines) {
    const clean = line.trim();
    if (!clean) continue;

    if (headers.some((header) => clean.toLowerCase().startsWith(header))) {
      capture = true;
      continue;
    }

    if (capture && /^(requirements|responsibilities|qualifications|benefits|about|회사|업무|자격|복지)/i.test(clean)) {
      break;
    }

    if (capture) {
      section.push(clean);
      if (section.join(" ").length > 220) break;
    }
  }

  return section.join(" ");
}

function isHeaderLine(line) {
  return /^(requirements|responsibilities|qualifications|benefits|about|company|who we are|role|position|job|salary|compensation|location|apply|how to apply|what you'll do|what you will do|업무|자격|복지|회사)/i.test(line.trim());
}

function extractSectionLines(text, headers) {
  const lines = text.split("\n");
  let capture = false;
  const section = [];

  for (const line of lines) {
    const clean = line.trim();
    if (!clean) continue;

    if (headers.some((header) => clean.toLowerCase().startsWith(header))) {
      capture = true;
      continue;
    }

    if (capture && isHeaderLine(clean)) {
      break;
    }

    if (capture) {
      section.push(clean);
      if (section.join(" ").length > 320) break;
    }
  }

  return section;
}

function extractFirstSentence(text) {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return "";
  const split = clean.split(/(?<=[.!?])\s/);
  return split[0] || clean;
}

function extractSentences(text, maxSentences) {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return "";
  const split = clean.split(/(?<=[.!?])\s/);
  return split.slice(0, maxSentences).join(" ");
}

function stripBullet(line) {
  return line.replace(/^[-*•\d+.]+\s*/, "").trim();
}

function summarizeFromLines(lines, maxItems) {
  const bullets = lines.filter((line) => /^[-*•\d+.]/.test(line.trim()));
  if (bullets.length > 0) {
    return bullets.slice(0, maxItems).map(stripBullet).join("; ");
  }
  return extractSentences(lines.join(" "), Math.min(2, maxItems));
}

function extractIntroParagraph(text) {
  const lines = text.split("\n");
  const intro = [];
  for (const line of lines) {
    const clean = line.trim();
    if (!clean) continue;
    if (isHeaderLine(clean)) break;
    intro.push(clean);
    if (intro.join(" ").length > 240) break;
  }
  return intro.join(" ");
}

function extractRole(text) {
  const hint = pickFromHints(text, roleHints);
  if (hint) return hint;
  const titleLine = text.split("\n").find((line) => /engineer|designer|manager|marketer|developer|researcher|analyst|product/i.test(line));
  return titleLine ? titleLine.trim() : "";
}

function extractSkills(text) {
  const found = new Set();
  const lowered = text.toLowerCase();
  for (const keyword of skillKeywords) {
    if (lowered.includes(keyword)) {
      found.add(keyword);
    }
  }
  return Array.from(found);
}

function extractCompensation(text) {
  for (const hint of compensationHints) {
    const match = text.match(hint);
    if (match) {
      return match[0].trim();
    }
  }
  return "";
}

function extractWorkMode(text) {
  for (const mode of workModeHints) {
    if (mode.re.test(text)) return mode.label;
  }
  return "";
}

function extractCompanySummary(text) {
  const aboutLines = extractSectionLines(text, [
    "about",
    "company",
    "who we are",
    "about us",
    "our mission",
    "what we do",
    "회사 소개",
    "회사",
    "우리는",
  ]);
  if (aboutLines.length) return summarizeFromLines(aboutLines, 2);
  const intro = extractIntroParagraph(text);
  if (intro) return extractSentences(intro, 2);
  return extractFirstSentence(text);
}

function extractHiringFor(text) {
  const respLines = extractSectionLines(text, [
    "responsibilities",
    "what you'll do",
    "what you will do",
    "role",
    "the role",
    "position",
    "your impact",
    "업무",
    "주요 업무",
  ]);
  if (respLines.length) return summarizeFromLines(respLines, 3);

  const sentenceMatch = text.match(/(.{0,120}(you will|responsible for|we are looking for|we're looking for|seeking).{0,120})/i);
  if (sentenceMatch && sentenceMatch[0]) return extractFirstSentence(sentenceMatch[0]);
  return extractFirstSentence(text);
}

function extractTags(skills) {
  const tags = skills.slice(0, 6);
  return tags;
}

function computeFingerprint(data) {
  const core = normalize(`${data.company} ${data.role} ${data.location}`);
  if (core.length > 4) return core;
  return normalize((data.jdText || "").slice(0, 180));
}

function findDuplicates(data) {
  const fingerprint = computeFingerprint(data);
  if (!fingerprint) return [];
  return entries.filter((entry) => entry.fingerprint === fingerprint);
}

function renderStats() {
  const total = entries.length;
  const applied = entries.filter((e) => e.status === "Applied").length;
  const interviewing = entries.filter((e) => e.status === "Interviewing").length;
  const offers = entries.filter((e) => e.status === "Offer").length;

  statsEl.innerHTML = [
    { label: "Total", value: total },
    { label: "Applied", value: applied },
    { label: "Interviewing", value: interviewing },
    { label: "Offers", value: offers },
  ]
    .map((stat) => `<div class="stat">${stat.label}: ${stat.value}</div>`)
    .join("");
}

function renderEntries() {
  const query = normalize(searchInput.value);
  const statusValue = statusFilter.value;
  const modeValue = modeFilter.value;

  const filtered = entries.filter((entry) => {
    const haystack = normalize(
      [
        entry.company,
        entry.role,
        entry.location,
        entry.skills.join(" "),
        entry.tags.join(" "),
      ].join(" ")
    );

    if (query && !haystack.includes(query)) return false;
    if (statusValue && entry.status !== statusValue) return false;
    if (modeValue && entry.workMode !== modeValue) return false;
    return true;
  });

  if (filtered.length === 0) {
    entriesEl.innerHTML = "<p>No entries yet. Paste a JD to get started.</p>";
    return;
  }

  entriesEl.innerHTML = filtered
    .map((entry) => {
      const meta = [entry.location, entry.workMode, entry.deadline ? `Deadline: ${entry.deadline}` : null]
        .filter(Boolean)
        .join(" • ");
      const badges = entry.skills
        .slice(0, 6)
        .map((skill) => `<span class="badge">${skill}</span>`)
        .join("");

      return `
        <div class="entry">
          <div class="entry-header">
            <div>
              <div class="entry-title">${entry.company || "Untitled company"} — ${entry.role || "Role"}</div>
              <div class="entry-meta">${meta}</div>
            </div>
            <div class="entry-actions">
              <select data-action="status" data-id="${entry.id}">
                ${["Not Applied", "Applied", "Interviewing", "Offer", "Rejected"]
                  .map((status) => `<option value="${status}" ${entry.status === status ? "selected" : ""}>${status}</option>`)
                  .join("")}
              </select>
              <button class="ghost" data-action="edit" data-id="${entry.id}">Edit</button>
              <button class="ghost" data-action="delete" data-id="${entry.id}">Delete</button>
            </div>
          </div>
          <div class="entry-meta">${entry.companySummary || ""}</div>
          <div class="entry-meta">Hiring for: ${entry.hiringFor || ""}</div>
          <div class="entry-meta">Compensation: ${entry.compensation || ""}</div>
          <div class="badges">${badges}</div>
        </div>
      `;
    })
    .join("");
}

function fillForm(data) {
  companyInput.value = data.company || "";
  locationInput.value = data.location || "";
  roleInput.value = data.role || "";
  workModeInput.value = data.workMode || "";
  compInput.value = data.compensation || "";
  companySummaryInput.value = data.companySummary || "";
  hiringInput.value = data.hiringFor || "";
  skillsInput.value = data.skills.join(", ");
  tagsInput.value = data.tags.join(", ");
  statusInput.value = data.status || "Not Applied";
  deadlineInput.value = data.deadline || "";
}

function clearForm() {
  editingId = null;
  fillForm({
    company: "",
    location: "",
    role: "",
    workMode: "",
    compensation: "",
    companySummary: "",
    hiringFor: "",
    skills: [],
    tags: [],
    status: "Not Applied",
    deadline: "",
  });
}

function parseJD(text) {
  const company = pickFromHints(text, companyHints);
  const location = pickFromHints(text, locationHints);
  const role = extractRole(text);
  const skills = extractSkills(text);
  const compensation = extractCompensation(text);
  const workMode = extractWorkMode(text);
  const companySummary = extractCompanySummary(text);
  const hiringFor = extractHiringFor(text);
  const tags = extractTags(skills);

  return {
    company,
    location,
    role,
    workMode,
    compensation,
    companySummary,
    hiringFor,
    skills,
    tags,
  };
}

function normalizeAIResult(payload) {
  return {
    company: payload.company || "",
    location: payload.location || "",
    role: payload.role || "",
    workMode: payload.workMode || "",
    compensation: payload.compensation || "",
    companySummary: payload.companySummary || "",
    hiringFor: payload.hiringFor || "",
    skills: Array.isArray(payload.skills) ? payload.skills : [],
    tags: Array.isArray(payload.tags) ? payload.tags : [],
  };
}

async function analyzeWithAI(text) {
  const response = await fetch("/summarize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jdText: text }),
  });

  if (!response.ok) {
    const details = await response.text();
    throw new Error(details || "AI request failed");
  }

  const data = await response.json();
  return normalizeAIResult(data.data || {});
}

analyzeBtn.addEventListener("click", async () => {
  const text = jdInput.value.trim();
  if (!text || isAnalyzing) return;

  isAnalyzing = true;
  analyzeBtn.disabled = true;
  aiStatus.textContent = "Summarizing with AI...";

  let parsed;
  try {
    parsed = await analyzeWithAI(text);
    aiStatus.textContent = "AI summary complete.";
  } catch (error) {
    parsed = parseJD(text);
    aiStatus.textContent = "AI unavailable. Used local rules instead.";
  }

  fillForm({
    ...parsed,
    status: statusInput.value || "Not Applied",
    deadline: deadlineInput.value || "",
  });

  const duplicates = findDuplicates({
    ...parsed,
    jdText: text,
  });

  if (duplicates.length > 0) {
    duplicateWarning.hidden = false;
    duplicateWarning.textContent = `Possible duplicate found (${duplicates.length}). Check the tracker below before saving.`;
  } else {
    duplicateWarning.hidden = true;
  }

  analyzeBtn.disabled = false;
  isAnalyzing = false;
});

clearBtn.addEventListener("click", () => {
  jdInput.value = "";
  duplicateWarning.hidden = true;
});

resetBtn.addEventListener("click", () => {
  clearForm();
});

entryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = {
    id: editingId || crypto.randomUUID(),
    createdAt: editingId ? entries.find((e) => e.id === editingId)?.createdAt : new Date().toISOString(),
    jdText: jdInput.value.trim(),
    company: companyInput.value.trim(),
    location: locationInput.value.trim(),
    role: roleInput.value.trim(),
    workMode: workModeInput.value,
    compensation: compInput.value.trim(),
    companySummary: companySummaryInput.value.trim(),
    hiringFor: hiringInput.value.trim(),
    skills: skillsInput.value.split(",").map((s) => s.trim()).filter(Boolean),
    tags: tagsInput.value.split(",").map((s) => s.trim()).filter(Boolean),
    status: statusInput.value,
    deadline: deadlineInput.value,
  };

  data.fingerprint = computeFingerprint(data);

  if (editingId) {
    entries = entries.map((entry) => (entry.id === editingId ? data : entry));
  } else {
    entries.unshift(data);
  }

  saveEntries();
  renderStats();
  renderEntries();
  clearForm();
  jdInput.value = "";
  duplicateWarning.hidden = true;

  await runSync(() => upsertEntryToSupabase(data));
});

entriesEl.addEventListener("click", async (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  const id = button.dataset.id;
  const action = button.dataset.action;
  const entry = entries.find((item) => item.id === id);
  if (!entry) return;

  if (action === "edit") {
    editingId = id;
    jdInput.value = entry.jdText || "";
    fillForm(entry);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  if (action === "delete") {
    entries = entries.filter((item) => item.id !== id);
    saveEntries();
    renderStats();
    renderEntries();
    await runSync(() => deleteEntryFromSupabase(id));
  }
});

entriesEl.addEventListener("change", async (event) => {
  const select = event.target.closest("select");
  if (!select || select.dataset.action !== "status") return;
  const id = select.dataset.id;
  entries = entries.map((entry) =>
    entry.id === id ? { ...entry, status: select.value } : entry
  );
  saveEntries();
  renderStats();

  const updated = entries.find((entry) => entry.id === id);
  if (updated) {
    await runSync(() => upsertEntryToSupabase(updated));
  }
});

searchInput.addEventListener("input", renderEntries);
statusFilter.addEventListener("change", renderEntries);
modeFilter.addEventListener("change", renderEntries);

exportBtn.addEventListener("click", () => {
  if (entries.length === 0) return;

  const headers = [
    "Company",
    "Role",
    "Location",
    "Work Mode",
    "Compensation",
    "Company Summary",
    "Hiring For",
    "Skills",
    "Tags",
    "Status",
    "Deadline",
    "Created At",
  ];

  const rows = entries.map((entry) => [
    entry.company,
    entry.role,
    entry.location,
    entry.workMode,
    entry.compensation,
    entry.companySummary,
    entry.hiringFor,
    entry.skills.join("; "),
    entry.tags.join("; "),
    entry.status,
    entry.deadline,
    entry.createdAt,
  ]);

  const csv = [headers, ...rows]
    .map((row) => row.map((cell) => `"${(cell || "").replace(/"/g, '""')}"`).join(","))
    .join("\n");

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "job-tracker.csv";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
});

clearAllBtn.addEventListener("click", async () => {
  const confirmed = window.confirm("Delete all entries? This cannot be undone.");
  if (!confirmed) return;
  entries = [];
  saveEntries();
  renderStats();
  renderEntries();
  await runSync(() => deleteAllEntriesFromSupabase());
});

async function initializeAuth() {
  if (!hasSupabase()) return;

  const { data } = await supabaseClient.auth.getSession();
  currentUser = data.session?.user || null;
  updateAuthUI();

  supabaseClient.auth.onAuthStateChange((_event, session) => {
    currentUser = session?.user || null;
    updateAuthUI();
    if (currentUser) {
      loadFromSupabase(true);
    }
  });
}

async function loadFromSupabase(shouldPrompt) {
  if (!hasSupabase() || !hasUser()) return;
  updateSyncStatus("Sync: loading...");
  const localEntries = entries.slice();

  try {
    if (shouldPrompt && localEntries.length > 0) {
      const key = `jobTrackerMigrationPrompted_${currentUser.id}`;
      let shouldUpload = false;
      if (localStorage.getItem(key) !== "1") {
        shouldUpload = window.confirm(
          "Upload your local entries to this account for sync?"
        );
        localStorage.setItem(key, "1");
      }
      if (shouldUpload) {
        await upsertEntriesToSupabase(localEntries);
      }
    } else if (localEntries.length > 0) {
      await upsertEntriesToSupabase(localEntries);
    }

    const remoteEntries = await fetchEntriesFromSupabase();
    if (localEntries.length > 0) {
      entries = mergeEntries(localEntries, remoteEntries);
    } else {
      entries = remoteEntries;
    }
    saveEntries();
    renderStats();
    renderEntries();
    updateSyncStatus("Sync: up to date");
  } catch (error) {
    console.warn("Failed to load from Supabase", error);
    updateSyncStatus("Sync: error (local only)");
  }
}

if (navAuthBtn) {
  navAuthBtn.addEventListener("click", async () => {
    if (!hasSupabase()) return;
    if (hasUser()) {
      await supabaseClient.auth.signOut();
      currentUser = null;
      updateAuthUI();
    } else {
      window.location.href = "auth.html";
    }
  });
}

async function init() {
  updateAuthUI();
  renderStats();
  renderEntries();
  clearForm();
  await initializeAuth();
  if (hasSupabase() && hasUser()) {
    await loadFromSupabase(false);
  }
}

init();
