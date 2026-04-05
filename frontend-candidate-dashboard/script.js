const candidates = [
  {
    id: 1,
    name: "Maya Chen",
    role: "Frontend Developer",
    status: "Interview",
    experience: 2,
    match: 92,
    location: "Toronto, ON",
    skills: ["React", "TypeScript", "CSS", "Accessibility"],
    summary:
      "Strong UI implementation skills with polished responsive work and clear component structure.",
    highlights: [
      "Built a dashboard with reusable charts and filters",
      "Improved Lighthouse accessibility score to 96",
      "Comfortable with design-system thinking",
    ],
  },
  {
    id: 2,
    name: "Noah Patel",
    role: "Frontend Engineer",
    status: "Review",
    experience: 3,
    match: 88,
    location: "Vancouver, BC",
    skills: ["JavaScript", "React", "Node", "Testing"],
    summary:
      "Balanced product-minded frontend engineer with strong debugging habits and practical full-stack support.",
    highlights: [
      "Implemented API-driven search and filtering UX",
      "Added loading and error states across user flows",
      "Improved bundle structure for feature pages",
    ],
  },
  {
    id: 3,
    name: "Ava Singh",
    role: "UI Developer",
    status: "Offer",
    experience: 1,
    match: 95,
    location: "Calgary, AB",
    skills: ["HTML", "CSS", "Figma", "Animation"],
    summary:
      "Strong eye for layout, interaction detail, and visual consistency across marketing and product screens.",
    highlights: [
      "Translated mockups into production-ready screens",
      "Created motion patterns for onboarding UI",
      "Maintained component naming consistency",
    ],
  },
  {
    id: 4,
    name: "Lucas Romero",
    role: "React Developer",
    status: "Interview",
    experience: 4,
    match: 90,
    location: "Montreal, QC",
    skills: ["React", "Next.js", "API Integration", "Tailwind"],
    summary:
      "Confident in shipping frontend features tied to API workflows, forms, and account experiences.",
    highlights: [
      "Shipped multi-step checkout experiences",
      "Integrated third-party APIs with graceful fallbacks",
      "Led frontend cleanup for legacy screens",
    ],
  },
  {
    id: 5,
    name: "Sofia Kim",
    role: "Junior Frontend Developer",
    status: "New",
    experience: 1,
    match: 84,
    location: "Edmonton, AB",
    skills: ["JavaScript", "HTML", "CSS", "Git"],
    summary:
      "Solid entry-level profile with careful implementation habits and a strong sense for improving UI clarity.",
    highlights: [
      "Built responsive landing pages from scratch",
      "Handled browser layout bugs methodically",
      "Communicated progress clearly in team settings",
    ],
  },
  {
    id: 6,
    name: "Ethan Brooks",
    role: "Product Frontend Engineer",
    status: "Review",
    experience: 5,
    match: 86,
    location: "Remote",
    skills: ["React", "Data Viz", "Performance", "Design Systems"],
    summary:
      "Experienced in turning product requirements into polished dashboards with careful frontend performance work.",
    highlights: [
      "Built analytics interfaces with dense data views",
      "Reduced unnecessary rerenders in reporting pages",
      "Documented frontend patterns for the team",
    ],
  },
];

const statusConfig = ["All", "New", "Review", "Interview", "Offer"];

const state = {
  selectedStatus: "All",
  searchTerm: "",
  sortBy: "match",
  selectedCandidateId: candidates[0].id,
};

const statusFilters = document.getElementById("statusFilters");
const statsGrid = document.getElementById("statsGrid");
const candidateList = document.getElementById("candidateList");
const detailPanel = document.getElementById("detailPanel");
const searchInput = document.getElementById("searchInput");
const sortSelect = document.getElementById("sortSelect");
const showTopMatches = document.getElementById("showTopMatches");
const resetFilters = document.getElementById("resetFilters");

function createStatusFilters() {
  statusFilters.innerHTML = "";

  statusConfig.forEach((status) => {
    const button = document.createElement("button");
    button.className = "filter-chip";
    button.textContent = status;
    button.type = "button";

    if (state.selectedStatus === status) {
      button.classList.add("is-active");
    }

    button.addEventListener("click", () => {
      state.selectedStatus = status;
      renderDashboard();
    });

    statusFilters.appendChild(button);
  });
}

function getFilteredCandidates() {
  const normalizedSearch = state.searchTerm.trim().toLowerCase();

  const filtered = candidates.filter((candidate) => {
    const matchesStatus =
      state.selectedStatus === "All" || candidate.status === state.selectedStatus;

    const matchesSearch =
      !normalizedSearch ||
      [candidate.name, candidate.role, candidate.location, ...candidate.skills]
        .join(" ")
        .toLowerCase()
        .includes(normalizedSearch);

    return matchesStatus && matchesSearch;
  });

  const sorted = [...filtered].sort((left, right) => {
    if (state.sortBy === "experience") {
      return right.experience - left.experience;
    }

    if (state.sortBy === "name") {
      return left.name.localeCompare(right.name);
    }

    return right.match - left.match;
  });

  return sorted;
}

function renderStats(items) {
  const averageMatch =
    items.length > 0
      ? Math.round(items.reduce((sum, candidate) => sum + candidate.match, 0) / items.length)
      : 0;

  const interviewCount = items.filter((candidate) => candidate.status === "Interview").length;
  const offerCount = items.filter((candidate) => candidate.status === "Offer").length;

  const stats = [
    { label: "Visible Candidates", value: items.length },
    { label: "Average Match", value: `${averageMatch}%` },
    { label: "Interview Stage", value: interviewCount },
    { label: "Offers Ready", value: offerCount },
  ];

  statsGrid.innerHTML = stats
    .map(
      (item) => `
        <article class="stats-card">
          <p class="stats-label">${item.label}</p>
          <p class="stats-value">${item.value}</p>
        </article>
      `
    )
    .join("");
}

function renderCandidateList(items) {
  if (!items.length) {
    candidateList.innerHTML = `
      <div class="empty-state">
        No candidates match the current filters. Try a different search term or reset the view.
      </div>
    `;
    return;
  }

  candidateList.innerHTML = items
    .map(
      (candidate) => `
        <article class="candidate-card ${
          state.selectedCandidateId === candidate.id ? "is-selected" : ""
        }" data-id="${candidate.id}">
          <div class="candidate-head">
            <div>
              <p class="candidate-name">${candidate.name}</p>
              <p class="candidate-role">${candidate.role}</p>
            </div>
            <div class="score-pill">${candidate.match}% match</div>
          </div>

          <div class="candidate-meta">
            <span class="meta-chip">${candidate.status}</span>
            <span class="meta-chip">${candidate.experience} yrs</span>
            <span class="meta-chip">${candidate.location}</span>
          </div>

          <div class="candidate-skills">
            ${candidate.skills.map((skill) => `<span class="skill-chip">${skill}</span>`).join("")}
          </div>

          <p class="candidate-note">${candidate.summary}</p>
        </article>
      `
    )
    .join("");

  candidateList.querySelectorAll(".candidate-card").forEach((card) => {
    card.addEventListener("click", () => {
      state.selectedCandidateId = Number(card.dataset.id);
      renderDashboard();
    });
  });
}

function renderDetail(items) {
  const selectedCandidate =
    items.find((candidate) => candidate.id === state.selectedCandidateId) || items[0];

  if (!selectedCandidate) {
    detailPanel.innerHTML = `
      <p class="detail-kicker">Candidate Detail</p>
      <h3>No selection</h3>
      <p class="detail-copy">Adjust filters or reset the view to inspect a candidate.</p>
    `;
    return;
  }

  state.selectedCandidateId = selectedCandidate.id;

  detailPanel.innerHTML = `
    <p class="detail-kicker">Candidate Detail</p>
    <h3>${selectedCandidate.name}</h3>
    <p class="detail-copy">${selectedCandidate.role} based in ${selectedCandidate.location} with
      ${selectedCandidate.experience} years of experience and a ${selectedCandidate.match}% fit score.</p>

    <div class="detail-block">
      <h4>Status</h4>
      <p>${selectedCandidate.status}</p>
    </div>

    <div class="detail-block">
      <h4>Top Skills</h4>
      <div class="candidate-skills">
        ${selectedCandidate.skills.map((skill) => `<span class="skill-chip">${skill}</span>`).join("")}
      </div>
    </div>

    <div class="detail-block">
      <h4>Summary</h4>
      <p>${selectedCandidate.summary}</p>
    </div>

    <div class="detail-block">
      <h4>Highlights</h4>
      <ul class="detail-list">
        ${selectedCandidate.highlights.map((item) => `<li>${item}</li>`).join("")}
      </ul>
    </div>
  `;
}

function renderDashboard() {
  const filteredCandidates = getFilteredCandidates();
  createStatusFilters();
  renderStats(filteredCandidates);
  renderCandidateList(filteredCandidates);
  renderDetail(filteredCandidates);
}

searchInput.addEventListener("input", (event) => {
  state.searchTerm = event.target.value;
  renderDashboard();
});

sortSelect.addEventListener("change", (event) => {
  state.sortBy = event.target.value;
  renderDashboard();
});

showTopMatches.addEventListener("click", () => {
  state.selectedStatus = "All";
  state.sortBy = "match";
  state.searchTerm = "";
  searchInput.value = "";
  sortSelect.value = "match";
  renderDashboard();
});

resetFilters.addEventListener("click", () => {
  state.selectedStatus = "All";
  state.searchTerm = "";
  state.sortBy = "match";
  state.selectedCandidateId = candidates[0].id;
  searchInput.value = "";
  sortSelect.value = "match";
  renderDashboard();
});

renderDashboard();
