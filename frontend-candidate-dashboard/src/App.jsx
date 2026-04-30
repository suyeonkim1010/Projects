import { useMemo, useRef, useState } from "react";

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
  {
    id: 7,
    name: "Priya Nair",
    role: "Frontend Platform Engineer",
    status: "Review",
    experience: 6,
    match: 89,
    location: "Ottawa, ON",
    skills: ["React", "TypeScript", "Design Systems", "Testing"],
    summary:
      "Strong platform-minded frontend engineer with experience building reusable UI primitives and team-wide standards.",
    highlights: [
      "Built shared component libraries used across multiple product teams",
      "Improved release confidence with stronger frontend test coverage",
      "Partnered with designers on token and pattern consistency",
    ],
  },
  {
    id: 8,
    name: "Jasper Lee",
    role: "Frontend Developer",
    status: "New",
    experience: 2,
    match: 82,
    location: "Winnipeg, MB",
    skills: ["JavaScript", "React", "CSS", "Accessibility"],
    summary:
      "Implementation-focused frontend developer with solid accessibility habits and careful UI polish.",
    highlights: [
      "Built accessible form flows for internal admin screens",
      "Improved keyboard navigation across interactive tables",
      "Worked closely with QA to resolve UI regressions quickly",
    ],
  },
  {
    id: 9,
    name: "Elena Martinez",
    role: "Growth Frontend Engineer",
    status: "Offer",
    experience: 4,
    match: 91,
    location: "Remote",
    skills: ["React", "Experimentation", "Analytics", "Performance"],
    summary:
      "Comfortable with growth-focused frontend work tied to experiment speed, analytics, and landing-page performance.",
    highlights: [
      "Ran A/B test implementations across acquisition funnels",
      "Improved Core Web Vitals on campaign pages",
      "Translated product metrics into UI experimentation priorities",
    ],
  },
  {
    id: 10,
    name: "Hana Park",
    role: "Design Systems Engineer",
    status: "Review",
    experience: 5,
    match: 87,
    location: "Halifax, NS",
    skills: ["React", "Storybook", "Tokens", "CSS"],
    summary:
      "Specialized in creating maintainable design-system foundations and reusable component APIs.",
    highlights: [
      "Shipped documented component patterns for internal teams",
      "Standardized spacing and typography tokens across product surfaces",
      "Reduced one-off UI implementation drift",
    ],
  },
  {
    id: 11,
    name: "Marcus Johnson",
    role: "Frontend Engineer",
    status: "New",
    experience: 3,
    match: 85,
    location: "Victoria, BC",
    skills: ["TypeScript", "React", "API Integration", "Playwright"],
    summary:
      "Balanced product engineer with solid ownership over connected UI flows and frontend testing.",
    highlights: [
      "Implemented account-management screens tied to API workflows",
      "Added Playwright coverage for critical checkout and settings flows",
      "Improved frontend error-state consistency across screens",
    ],
  },
  {
    id: 12,
    name: "Amelia Green",
    role: "Junior UI Engineer",
    status: "Offer",
    experience: 1,
    match: 80,
    location: "Quebec City, QC",
    skills: ["HTML", "CSS", "Figma", "React"],
    summary:
      "Early-career UI engineer with strong visual execution, handoff discipline, and responsive layout habits.",
    highlights: [
      "Built marketing and product support pages from Figma handoff",
      "Handled responsive fixes across tablet and mobile breakpoints",
      "Maintained visual consistency across repeated sections",
    ],
  },
  {
    id: 13,
    name: "Daniel Wu",
    role: "Senior Frontend Engineer",
    status: "Interview",
    experience: 7,
    match: 93,
    location: "Toronto, ON",
    skills: ["React", "TypeScript", "Performance", "Design Systems"],
    summary:
      "Experienced frontend engineer with strong ownership over complex UI architecture and high-traffic product surfaces.",
    highlights: [
      "Led refactors for reusable frontend architecture across product teams",
      "Reduced render bottlenecks in dashboard-heavy pages",
      "Mentored teammates on component composition and DX",
    ],
  },
  {
    id: 14,
    name: "Nina Desai",
    role: "Frontend Accessibility Engineer",
    status: "Review",
    experience: 4,
    match: 89,
    location: "Vancouver, BC",
    skills: ["Accessibility", "React", "CSS", "Testing"],
    summary:
      "Focused on accessible interaction patterns, semantic markup, and frontend quality improvements.",
    highlights: [
      "Audited major user flows for keyboard and screen-reader support",
      "Introduced accessibility regression checks into review workflows",
      "Improved form semantics across customer-facing screens",
    ],
  },
  {
    id: 15,
    name: "Oliver Grant",
    role: "Frontend Performance Engineer",
    status: "New",
    experience: 5,
    match: 88,
    location: "Remote",
    skills: ["React", "Performance", "Profiling", "JavaScript"],
    summary:
      "Strong at profiling, frontend optimization, and improving interaction speed for busy application surfaces.",
    highlights: [
      "Lowered interaction latency on data-heavy views",
      "Built profiling playbooks for product teams",
      "Improved chunking strategy for route-level code splits",
    ],
  },
  {
    id: 16,
    name: "Chloe Martin",
    role: "Product UI Engineer",
    status: "Offer",
    experience: 3,
    match: 90,
    location: "Montreal, QC",
    skills: ["React", "Figma", "Component Design", "CSS"],
    summary:
      "Comfortable translating product direction into polished UI systems with strong implementation consistency.",
    highlights: [
      "Converted design explorations into stable reusable components",
      "Improved consistency across navigation and form patterns",
      "Worked closely with PM and design on iteration quality",
    ],
  },
  {
    id: 17,
    name: "Leo Carter",
    role: "Frontend Developer",
    status: "Review",
    experience: 2,
    match: 83,
    location: "Edmonton, AB",
    skills: ["JavaScript", "React", "Git", "Playwright"],
    summary:
      "Practical frontend developer with steady delivery habits and clear focus on UI reliability.",
    highlights: [
      "Added Playwright coverage for account and settings flows",
      "Improved bug turnaround through clearer issue reproduction",
      "Built simple internal dashboards with reusable filters",
    ],
  },
  {
    id: 18,
    name: "Grace Miller",
    role: "Design Systems Frontend Engineer",
    status: "Interview",
    experience: 6,
    match: 91,
    location: "Halifax, NS",
    skills: ["Storybook", "React", "Tokens", "Accessibility"],
    summary:
      "Strong in shared frontend systems, documentation, and keeping multi-team UI work aligned.",
    highlights: [
      "Maintained shared component docs for engineering and design",
      "Standardized token usage across core product surfaces",
      "Improved accessibility defaults in the design system",
    ],
  },
  {
    id: 19,
    name: "Mason Alvarez",
    role: "Growth UI Developer",
    status: "New",
    experience: 2,
    match: 81,
    location: "Ottawa, ON",
    skills: ["HTML", "CSS", "Experimentation", "Analytics"],
    summary:
      "Focused on growth-oriented UI work, rapid experimentation, and fast turnaround for campaign experiences.",
    highlights: [
      "Built and iterated landing pages tied to acquisition goals",
      "Tracked experiment outcomes with clear implementation notes",
      "Improved visual consistency under short release cycles",
    ],
  },
  {
    id: 20,
    name: "Sara Thompson",
    role: "Junior React Developer",
    status: "Offer",
    experience: 1,
    match: 79,
    location: "Winnipeg, MB",
    skills: ["React", "JavaScript", "CSS", "API Integration"],
    summary:
      "Early-career React developer with good product instincts and steady progress on connected frontend flows.",
    highlights: [
      "Built API-driven profile and settings views",
      "Handled form validation and error messaging carefully",
      "Worked through UI bugs with strong iteration discipline",
    ],
  },
  {
    id: 21,
    name: "Ivy Turner",
    role: "Frontend Developer",
    status: "Review",
    experience: 2,
    match: 84,
    location: "Toronto, ON",
    skills: ["React", "CSS", "JavaScript", "Accessibility"],
    summary: "Careful UI implementer with strong attention to accessibility and responsive behavior.",
    highlights: [
      "Shipped reusable card and table components",
      "Improved keyboard focus handling across forms",
      "Worked through visual QA issues quickly",
    ],
  },
  {
    id: 22,
    name: "Ryan Cooper",
    role: "React Engineer",
    status: "New",
    experience: 4,
    match: 86,
    location: "Vancouver, BC",
    skills: ["React", "TypeScript", "API Integration", "Testing"],
    summary: "Product-oriented engineer comfortable with connected flows and pragmatic frontend testing.",
    highlights: [
      "Built authenticated dashboard screens",
      "Handled API loading and error states cleanly",
      "Added test coverage for important account flows",
    ],
  },
  {
    id: 23,
    name: "Fatima Noor",
    role: "UI Engineer",
    status: "Offer",
    experience: 3,
    match: 88,
    location: "Calgary, AB",
    skills: ["HTML", "CSS", "React", "Figma"],
    summary: "Strong visual UI engineer with reliable handoff execution and component discipline.",
    highlights: [
      "Implemented polished responsive marketing modules",
      "Improved consistency between mockups and production screens",
      "Maintained shared UI documentation",
    ],
  },
  {
    id: 24,
    name: "Ben Carter",
    role: "Frontend Platform Developer",
    status: "Review",
    experience: 6,
    match: 90,
    location: "Remote",
    skills: ["React", "Design Systems", "TypeScript", "Storybook"],
    summary: "Platform-minded frontend developer focused on shared component quality and reuse.",
    highlights: [
      "Maintained internal component packages",
      "Standardized component APIs across teams",
      "Reduced duplication in product UI work",
    ],
  },
  {
    id: 25,
    name: "Emma Ross",
    role: "Junior Frontend Developer",
    status: "New",
    experience: 1,
    match: 78,
    location: "Edmonton, AB",
    skills: ["JavaScript", "HTML", "CSS", "Git"],
    summary: "Entry-level frontend candidate with steady implementation habits and good UI fundamentals.",
    highlights: [
      "Built landing pages with reusable sections",
      "Resolved browser layout issues across breakpoints",
      "Worked clearly from issue lists and feedback",
    ],
  },
  {
    id: 26,
    name: "Victor Nguyen",
    role: "Frontend Engineer",
    status: "Review",
    experience: 5,
    match: 87,
    location: "Ottawa, ON",
    skills: ["React", "Performance", "JavaScript", "Playwright"],
    summary: "Balanced engineer focused on UI performance, reliability, and maintainable product flows.",
    highlights: [
      "Improved responsiveness on data-heavy screens",
      "Added E2E coverage for critical purchase flows",
      "Cleaned up legacy feature flags in frontend code",
    ],
  },
  {
    id: 27,
    name: "Lila Ahmed",
    role: "Design Systems Engineer",
    status: "Offer",
    experience: 4,
    match: 89,
    location: "Montreal, QC",
    skills: ["React", "Tokens", "CSS", "Storybook"],
    summary: "Strong at building scalable component foundations and aligning design with implementation.",
    highlights: [
      "Maintained token usage across products",
      "Documented component behavior for teams",
      "Reduced visual inconsistency across pages",
    ],
  },
  {
    id: 28,
    name: "Aaron Bell",
    role: "Frontend Developer",
    status: "New",
    experience: 2,
    match: 82,
    location: "Winnipeg, MB",
    skills: ["React", "JavaScript", "CSS", "Testing"],
    summary: "Implementation-focused frontend developer with good instincts for clean user flows.",
    highlights: [
      "Built internal admin forms and dashboards",
      "Handled bug reproduction carefully",
      "Worked with QA on frontend regressions",
    ],
  },
  {
    id: 29,
    name: "Jade Wilson",
    role: "Growth Frontend Engineer",
    status: "Review",
    experience: 3,
    match: 85,
    location: "Halifax, NS",
    skills: ["React", "Analytics", "Experimentation", "Performance"],
    summary: "Comfortable working on growth surfaces tied to experimentation and conversion-focused UX.",
    highlights: [
      "Implemented multiple A/B experiment variants",
      "Improved page speed for acquisition journeys",
      "Tracked test outcomes with product analysts",
    ],
  },
  {
    id: 30,
    name: "Cole Ramirez",
    role: "Frontend Engineer",
    status: "Offer",
    experience: 5,
    match: 90,
    location: "Victoria, BC",
    skills: ["TypeScript", "React", "API Integration", "Performance"],
    summary: "Strong ownership over connected frontend systems and pragmatic delivery in product teams.",
    highlights: [
      "Built account and billing experiences",
      "Reduced rendering overhead in feature pages",
      "Improved API-state consistency across screens",
    ],
  },
  {
    id: 31,
    name: "Naomi Foster",
    role: "UI Developer",
    status: "Review",
    experience: 2,
    match: 83,
    location: "Quebec City, QC",
    skills: ["HTML", "CSS", "Animation", "Figma"],
    summary: "UI-focused developer with polished execution and attention to motion and spacing detail.",
    highlights: [
      "Implemented animated onboarding components",
      "Improved consistency in repeated UI blocks",
      "Built responsive sections from design handoff",
    ],
  },
  {
    id: 32,
    name: "Samuel Price",
    role: "React Frontend Engineer",
    status: "New",
    experience: 4,
    match: 86,
    location: "Toronto, ON",
    skills: ["React", "TypeScript", "Testing", "Accessibility"],
    summary: "Well-rounded React engineer with good judgment around testing and user-facing reliability.",
    highlights: [
      "Added regression coverage for navigation flows",
      "Improved semantic structure on shared screens",
      "Maintained component libraries with clear APIs",
    ],
  },
  {
    id: 33,
    name: "Zara Liu",
    role: "Frontend Product Engineer",
    status: "Offer",
    experience: 3,
    match: 88,
    location: "Remote",
    skills: ["React", "JavaScript", "API Integration", "Design Systems"],
    summary: "Strong product frontend engineer with a focus on feature clarity and practical system reuse.",
    highlights: [
      "Built reusable product detail modules",
      "Handled API state transitions for profile flows",
      "Aligned features with shared component patterns",
    ],
  },
  {
    id: 34,
    name: "Harper Stone",
    role: "Accessibility Engineer",
    status: "Review",
    experience: 5,
    match: 87,
    location: "Ottawa, ON",
    skills: ["Accessibility", "React", "Testing", "CSS"],
    summary: "Accessibility specialist who improves usability without slowing down frontend delivery.",
    highlights: [
      "Audited customer flows for screen-reader issues",
      "Improved focus management across dialogs",
      "Partnered with engineers on inclusive defaults",
    ],
  },
  {
    id: 35,
    name: "Tyler Evans",
    role: "Junior UI Engineer",
    status: "New",
    experience: 1,
    match: 77,
    location: "Edmonton, AB",
    skills: ["HTML", "CSS", "React", "Git"],
    summary: "Entry-level UI engineer with solid layout habits and willingness to iterate quickly.",
    highlights: [
      "Built responsive feature marketing pages",
      "Handled visual fixes across device sizes",
      "Worked from design review comments efficiently",
    ],
  },
  {
    id: 36,
    name: "Sophia Bennett",
    role: "Frontend Systems Engineer",
    status: "Review",
    experience: 6,
    match: 91,
    location: "Vancouver, BC",
    skills: ["React", "Storybook", "Tokens", "TypeScript"],
    summary: "Experienced systems engineer focused on reuse, consistency, and robust frontend foundations.",
    highlights: [
      "Scaled design-system usage across multiple apps",
      "Maintained typed component contracts",
      "Reduced duplicated interface logic",
    ],
  },
  {
    id: 37,
    name: "Gabriel Kim",
    role: "Frontend Engineer",
    status: "Offer",
    experience: 4,
    match: 89,
    location: "Calgary, AB",
    skills: ["React", "Playwright", "API Integration", "JavaScript"],
    summary: "Reliable engineer with good ownership over interactive frontend flows and release confidence.",
    highlights: [
      "Added Playwright checks for settings and auth flows",
      "Improved API error handling in user screens",
      "Refined detail-panel interaction states",
    ],
  },
  {
    id: 38,
    name: "Mila Kapoor",
    role: "Frontend Developer",
    status: "New",
    experience: 2,
    match: 81,
    location: "Remote",
    skills: ["React", "CSS", "Figma", "Accessibility"],
    summary: "Frontend developer with strong UI instincts and careful attention to polished interaction states.",
    highlights: [
      "Built reusable onboarding modules",
      "Improved contrast and keyboard behavior",
      "Translated design feedback into cleaner components",
    ],
  },
  {
    id: 39,
    name: "Jordan White",
    role: "Growth UI Engineer",
    status: "Review",
    experience: 3,
    match: 84,
    location: "Winnipeg, MB",
    skills: ["HTML", "CSS", "Analytics", "Experimentation"],
    summary: "Focused on rapid growth UI work with a good eye for clean, conversion-friendly execution.",
    highlights: [
      "Built campaign variants for acquisition pages",
      "Improved readability on high-traffic forms",
      "Worked closely with marketing and product teams",
    ],
  },
  {
    id: 40,
    name: "Lucy Powell",
    role: "React Engineer",
    status: "Offer",
    experience: 5,
    match: 90,
    location: "Halifax, NS",
    skills: ["React", "TypeScript", "Performance", "Testing"],
    summary: "Strong React engineer with good instincts for scale, clarity, and frontend release quality.",
    highlights: [
      "Refined dashboard behavior for large result sets",
      "Improved perceived performance in reporting pages",
      "Maintained stable UI behavior during product iteration",
    ],
  },
  {
    id: 41,
    name: "Evelyn Moore",
    role: "Product Frontend Engineer",
    status: "Review",
    experience: 4,
    match: 88,
    location: "Victoria, BC",
    skills: ["React", "Design Systems", "JavaScript", "API Integration"],
    summary: "Comfortable balancing feature delivery, product polish, and shared UI consistency.",
    highlights: [
      "Built profile and onboarding product surfaces",
      "Worked across design-system and feature teams",
      "Improved consistency in account-management screens",
    ],
  },
  {
    id: 42,
    name: "Connor Reed",
    role: "Frontend Engineer",
    status: "New",
    experience: 3,
    match: 83,
    location: "Montreal, QC",
    skills: ["React", "JavaScript", "CSS", "Git"],
    summary: "Steady frontend engineer with solid habits around implementation detail and team collaboration.",
    highlights: [
      "Built reusable search and filter interfaces",
      "Handled layout and spacing regressions cleanly",
      "Worked well with product feedback loops",
    ],
  },
  {
    id: 43,
    name: "Ariana Flores",
    role: "Design Systems Developer",
    status: "Offer",
    experience: 5,
    match: 89,
    location: "Toronto, ON",
    skills: ["React", "Storybook", "Tokens", "Accessibility"],
    summary: "Strong systems developer who improves consistency and accessibility across shared UI layers.",
    highlights: [
      "Documented component usage across engineering teams",
      "Built accessible defaults into component APIs",
      "Improved token adoption through better tooling",
    ],
  },
  {
    id: 44,
    name: "Theo Martin",
    role: "Frontend Developer",
    status: "Review",
    experience: 2,
    match: 82,
    location: "Ottawa, ON",
    skills: ["React", "CSS", "HTML", "Playwright"],
    summary: "Implementation-focused frontend developer with solid testing instincts and clean UI habits.",
    highlights: [
      "Added browser coverage for core product journeys",
      "Built reusable form and card patterns",
      "Improved consistency in UI review cycles",
    ],
  },
  {
    id: 45,
    name: "Holly Scott",
    role: "Junior Frontend Engineer",
    status: "New",
    experience: 1,
    match: 76,
    location: "Quebec City, QC",
    skills: ["JavaScript", "React", "CSS", "Accessibility"],
    summary: "Early-career engineer with strong UI fundamentals and good responsiveness to design feedback.",
    highlights: [
      "Built simple product support screens",
      "Handled responsive fixes with consistency",
      "Improved semantics in component markup",
    ],
  },
  {
    id: 46,
    name: "Riley Hughes",
    role: "Frontend Performance Engineer",
    status: "Review",
    experience: 6,
    match: 90,
    location: "Remote",
    skills: ["Performance", "React", "Profiling", "TypeScript"],
    summary: "Experienced performance-focused engineer who improves responsiveness and rendering stability.",
    highlights: [
      "Profiled rendering bottlenecks on dashboards",
      "Improved interaction speed on complex tables",
      "Guided teams on frontend performance practices",
    ],
  },
  {
    id: 47,
    name: "Megan Clarke",
    role: "Frontend Product Engineer",
    status: "Offer",
    experience: 4,
    match: 87,
    location: "Edmonton, AB",
    skills: ["React", "API Integration", "Testing", "Design Systems"],
    summary: "Product-focused engineer with balanced frontend delivery across interactive and data-driven screens.",
    highlights: [
      "Shipped account and dashboard improvements",
      "Improved testing around user-critical flows",
      "Kept new features aligned with shared patterns",
    ],
  },
  {
    id: 48,
    name: "Isaac Perez",
    role: "UI Engineer",
    status: "Review",
    experience: 3,
    match: 84,
    location: "Vancouver, BC",
    skills: ["HTML", "CSS", "Animation", "React"],
    summary: "UI engineer with strong visual execution and a good sense for motion and layout quality.",
    highlights: [
      "Built polished onboarding and help-center screens",
      "Improved visual consistency across reusable sections",
      "Handled micro-interaction updates carefully",
    ],
  },
  {
    id: 49,
    name: "Kylie Brooks",
    role: "React Frontend Developer",
    status: "New",
    experience: 2,
    match: 80,
    location: "Halifax, NS",
    skills: ["React", "JavaScript", "CSS", "API Integration"],
    summary: "Steady frontend contributor with good ownership over small-to-medium interactive product features.",
    highlights: [
      "Built API-connected forms and settings pages",
      "Improved empty and error states in user flows",
      "Maintained consistency with shared styles",
    ],
  },
  {
    id: 50,
    name: "Peter Shah",
    role: "Frontend Engineer",
    status: "Offer",
    experience: 5,
    match: 88,
    location: "Toronto, ON",
    skills: ["React", "TypeScript", "Playwright", "Performance"],
    summary: "Strong frontend engineer with practical ownership across UI quality, browser testing, and performance.",
    highlights: [
      "Added E2E coverage for critical customer journeys",
      "Improved render stability on feature-rich screens",
      "Collaborated closely across product and QA reviews",
    ],
  },
];

const statuses = ["All", "New", "Review", "Interview", "Offer"];
const regions = ["All", "Remote", "AB", "BC", "MB", "NS", "ON", "QC"];

function App() {
  const [selectedStatus, setSelectedStatus] = useState("All");
  const [selectedRegion, setSelectedRegion] = useState("All");
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState("match");
  const [selectedCandidateId, setSelectedCandidateId] = useState(candidates[0].id);
  const [actionMessage, setActionMessage] = useState("");
  const [isListHighlighted, setIsListHighlighted] = useState(false);
  const messageTimerRef = useRef(null);
  const highlightTimerRef = useRef(null);

  const flashResults = (message) => {
    setActionMessage(message);
    setIsListHighlighted(true);

    window.clearTimeout(messageTimerRef.current);
    window.clearTimeout(highlightTimerRef.current);

    messageTimerRef.current = window.setTimeout(() => {
      setActionMessage("");
    }, 1800);

    highlightTimerRef.current = window.setTimeout(() => {
      setIsListHighlighted(false);
    }, 900);
  };

  const filteredCandidates = useMemo(() => {
    const normalizedSearch = searchTerm.trim().toLowerCase();

    const filtered = candidates.filter((candidate) => {
      const matchesStatus = selectedStatus === "All" || candidate.status === selectedStatus;
      const matchesRegion =
        selectedRegion === "All" ||
        (selectedRegion === "Remote"
          ? candidate.location === "Remote"
          : candidate.location.endsWith(`, ${selectedRegion}`));
      const matchesSearch =
        !normalizedSearch ||
        [candidate.name, candidate.role, candidate.location, ...candidate.skills]
          .join(" ")
          .toLowerCase()
          .includes(normalizedSearch);

      return matchesStatus && matchesRegion && matchesSearch;
    });

    return [...filtered].sort((left, right) => {
      if (sortBy === "experience") return right.experience - left.experience;
      if (sortBy === "name") return left.name.localeCompare(right.name);
      return right.match - left.match;
    });
  }, [searchTerm, selectedStatus, selectedRegion, sortBy]);

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
    setSelectedRegion("All");
    setSearchTerm("");
    setSortBy("match");
    setSelectedCandidateId(candidates[0].id);
    flashResults("View reset to the default ranked candidate list.");
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
          <p className="filter-group-label">Status</p>
          <div className="filter-stack">
            {statuses.map((status) => (
              <button
                key={status}
                className={`filter-chip ${selectedStatus === status ? "is-active" : ""}`}
                type="button"
                data-testid={`status-filter-${status.toLowerCase()}`}
                onClick={() => setSelectedStatus(status)}
              >
                {status}
              </button>
            ))}
          </div>
          <p className="filter-group-label">Region</p>
          <div className="filter-stack region-stack">
            {regions.map((region) => (
              <button
                key={region}
                className={`filter-chip ${selectedRegion === region ? "is-active" : ""}`}
                type="button"
                data-testid={`region-filter-${region.toLowerCase().replace(/\s+/g, "-")}`}
                onClick={() => {
                  setSelectedRegion(region);
                  flashResults(
                    region === "All"
                      ? "Showing candidates from all locations."
                      : `Filtering candidates for ${region}.`
                  );
                }}
              >
                {region}
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
              data-testid="show-top-matches"
              onClick={() => {
                setSelectedStatus("All");
                setSortBy("match");
                setSearchTerm("");
                flashResults("Showing candidates ranked by highest match.");
              }}
            >
              Show Top Matches
            </button>
            <button className="ghost-btn" type="button" data-testid="reset-view" onClick={resetView}>
              Reset View
            </button>
          </div>
        </section>

        <section className="stats-grid">
          {stats.map((item) => (
            <article className="stats-card" key={item.label}>
              <p className="stats-label">{item.label}</p>
              <p
                className="stats-value"
                data-testid={item.label.toLowerCase().replace(/\s+/g, "-")}
              >
                {item.value}
              </p>
            </article>
          ))}
        </section>

        {actionMessage ? (
          <div className="action-banner" data-testid="action-banner">
            {actionMessage}
          </div>
        ) : null}

        <section className="toolbar">
          <label className="search-box">
            <span>Search</span>
            <input
              type="text"
              placeholder="Search by name, role, or skill"
              data-testid="candidate-search"
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
            />
          </label>

          <label className="select-box">
            <span>Sort</span>
            <select
              value={sortBy}
              data-testid="sort-select"
              onChange={(event) => setSortBy(event.target.value)}
            >
              <option value="match">Highest Match</option>
              <option value="experience">Most Experience</option>
              <option value="name">Name</option>
            </select>
          </label>
        </section>

        <section className={`content-grid ${isListHighlighted ? "is-refreshed" : ""}`}>
          <div className="candidate-list">
            {filteredCandidates.length ? (
              filteredCandidates.map((candidate) => (
                <article
                  key={candidate.id}
                  className={`candidate-card ${
                    selectedCandidate?.id === candidate.id ? "is-selected" : ""
                  }`}
                  data-testid={`candidate-card-${candidate.id}`}
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
              <div className="empty-state" data-testid="empty-state">
                No candidates match the current filters. Try a different search term or reset the view.
              </div>
            )}
          </div>

          <aside className="detail-panel" data-testid="candidate-detail-panel">
            {selectedCandidate ? (
              <>
                <p className="detail-kicker">Candidate Detail</p>
                <h3 data-testid="candidate-detail-name">{selectedCandidate.name}</h3>
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
