import { useMemo, useState } from "react";

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

const statuses = ["All", "New", "Review", "Interview", "Offer"];

function App() {
  const [selectedStatus, setSelectedStatus] = useState("All");
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState("match");
  const [selectedCandidateId, setSelectedCandidateId] = useState(candidates[0].id);

  const filteredCandidates = useMemo(() => {
    const normalizedSearch = searchTerm.trim().toLowerCase();

    const filtered = candidates.filter((candidate) => {
      const matchesStatus = selectedStatus === "All" || candidate.status === selectedStatus;
      const matchesSearch =
        !normalizedSearch ||
        [candidate.name, candidate.role, candidate.location, ...candidate.skills]
          .join(" ")
          .toLowerCase()
          .includes(normalizedSearch);

      return matchesStatus && matchesSearch;
    });

    return [...filtered].sort((left, right) => {
      if (sortBy === "experience") return right.experience - left.experience;
      if (sortBy === "name") return left.name.localeCompare(right.name);
      return right.match - left.match;
    });
  }, [searchTerm, selectedStatus, sortBy]);

  const selectedCandidate =
    filteredCandidates.find((candidate) => candidate.id === selectedCandidateId) ||
    filteredCandidates[0] ||
    null;

  const stats = [
    { label: "Visible Candidates", value: filteredCandidates.length },
    {
      label: "Average Match",
      value: filteredCandidates.length
        ? `${Math.round(
            filteredCandidates.reduce((sum, candidate) => sum + candidate.match, 0) /
              filteredCandidates.length
          )}%`
        : "0%",
    },
    {
      label: "Interview Stage",
      value: filteredCandidates.filter((candidate) => candidate.status === "Interview").length,
    },
    {
      label: "Offers Ready",
      value: filteredCandidates.filter((candidate) => candidate.status === "Offer").length,
    },
  ];

  const resetView = () => {
    setSelectedStatus("All");
    setSearchTerm("");
    setSortBy("match");
    setSelectedCandidateId(candidates[0].id);
  };

  return (
    <div className="page-shell">
      <aside className="sidebar">
        <p className="eyebrow">React Portfolio Demo</p>
        <h1>Candidate Dashboard</h1>
        <p className="sidebar-copy">
          A React dashboard concept focused on filtering, ranking, and reviewing candidates through a clean
          frontend workflow.
        </p>

        <div className="sidebar-panel">
          <p className="panel-label">Quick Filters</p>
          <div className="filter-stack">
            {statuses.map((status) => (
              <button
                key={status}
                className={`filter-chip ${selectedStatus === status ? "is-active" : ""}`}
                type="button"
                onClick={() => setSelectedStatus(status)}
              >
                {status}
              </button>
            ))}
          </div>
        </div>

        <div className="sidebar-panel">
          <p className="panel-label">View Focus</p>
          <ul className="focus-list">
            <li>Search candidates instantly</li>
            <li>Review skill match summaries</li>
            <li>Inspect candidate details without leaving the page</li>
          </ul>
        </div>
      </aside>

      <main className="main-content">
        <section className="hero-card">
          <div>
            <p className="eyebrow">Recruiting Workspace</p>
            <h2>Shortlist talent faster with ranked React workflows.</h2>
            <p className="hero-copy">
              Explore interactive filtering, live search, candidate detail views, and compact dashboard
              stats in one screen.
            </p>
          </div>
          <div className="hero-actions">
            <button
              className="primary-btn"
              type="button"
              onClick={() => {
                setSelectedStatus("All");
                setSortBy("match");
                setSearchTerm("");
              }}
            >
              Show Top Matches
            </button>
            <button className="ghost-btn" type="button" onClick={resetView}>
              Reset View
            </button>
          </div>
        </section>

        <section className="stats-grid">
          {stats.map((item) => (
            <article className="stats-card" key={item.label}>
              <p className="stats-label">{item.label}</p>
              <p className="stats-value">{item.value}</p>
            </article>
          ))}
        </section>

        <section className="toolbar">
          <label className="search-box">
            <span>Search</span>
            <input
              type="text"
              placeholder="Search by name, role, or skill"
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
            />
          </label>

          <label className="select-box">
            <span>Sort</span>
            <select value={sortBy} onChange={(event) => setSortBy(event.target.value)}>
              <option value="match">Highest Match</option>
              <option value="experience">Most Experience</option>
              <option value="name">Name</option>
            </select>
          </label>
        </section>

        <section className="content-grid">
          <div className="candidate-list">
            {filteredCandidates.length ? (
              filteredCandidates.map((candidate) => (
                <article
                  key={candidate.id}
                  className={`candidate-card ${
                    selectedCandidate?.id === candidate.id ? "is-selected" : ""
                  }`}
                  onClick={() => setSelectedCandidateId(candidate.id)}
                >
                  <div className="candidate-head">
                    <div>
                      <p className="candidate-name">{candidate.name}</p>
                      <p className="candidate-role">{candidate.role}</p>
                    </div>
                    <div className="score-pill">{candidate.match}% match</div>
                  </div>

                  <div className="candidate-meta">
                    <span className="meta-chip">{candidate.status}</span>
                    <span className="meta-chip">{candidate.experience} yrs</span>
                    <span className="meta-chip">{candidate.location}</span>
                  </div>

                  <div className="candidate-skills">
                    {candidate.skills.map((skill) => (
                      <span className="skill-chip" key={skill}>
                        {skill}
                      </span>
                    ))}
                  </div>

                  <p className="candidate-note">{candidate.summary}</p>
                </article>
              ))
            ) : (
              <div className="empty-state">
                No candidates match the current filters. Try a different search term or reset the view.
              </div>
            )}
          </div>

          <aside className="detail-panel">
            {selectedCandidate ? (
              <>
                <p className="detail-kicker">Candidate Detail</p>
                <h3>{selectedCandidate.name}</h3>
                <p className="detail-copy">
                  {selectedCandidate.role} based in {selectedCandidate.location} with{" "}
                  {selectedCandidate.experience} years of experience and a {selectedCandidate.match}% fit
                  score.
                </p>

                <div className="detail-block">
                  <h4>Status</h4>
                  <p>{selectedCandidate.status}</p>
                </div>

                <div className="detail-block">
                  <h4>Top Skills</h4>
                  <div className="candidate-skills">
                    {selectedCandidate.skills.map((skill) => (
                      <span className="skill-chip" key={skill}>
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="detail-block">
                  <h4>Summary</h4>
                  <p>{selectedCandidate.summary}</p>
                </div>

                <div className="detail-block">
                  <h4>Highlights</h4>
                  <ul className="detail-list">
                    {selectedCandidate.highlights.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              </>
            ) : (
              <>
                <p className="detail-kicker">Candidate Detail</p>
                <h3>No selection</h3>
                <p className="detail-copy">
                  Adjust filters or reset the view to inspect a candidate.
                </p>
              </>
            )}
          </aside>
        </section>
      </main>
    </div>
  );
}

export default App;
