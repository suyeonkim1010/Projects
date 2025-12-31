const authEmailInput = document.getElementById("authEmail");
const authPasswordInput = document.getElementById("authPassword");
const signUpBtn = document.getElementById("signUpBtn");
const signInBtn = document.getElementById("signInBtn");
const signOutBtn = document.getElementById("signOutBtn");
const authStatus = document.getElementById("authStatus");
const authGrid = document.getElementById("authGrid");

const SUPABASE_URL = "https://ecqhywxtmlriuqctwtsd.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVjcWh5d3h0bWxyaXVxY3R3dHNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjcxNDY2MDUsImV4cCI6MjA4MjcyMjYwNX0.32CEEDovRUzLUDwY-EHx3dZIgEPqjVkerxgxhD1W3XE";
const supabaseClient = window.supabase?.createClient?.(SUPABASE_URL, SUPABASE_ANON_KEY) || null;

let currentUser = null;

function updateAuthUI() {
  if (!authStatus) return;

  if (!supabaseClient) {
    authStatus.textContent = "Supabase not loaded. Refresh or check network.";
    return;
  }

  if (currentUser) {
    authStatus.textContent = `Signed in as ${currentUser.email}`;
    if (authGrid) authGrid.hidden = true;
    signUpBtn.hidden = true;
    signInBtn.hidden = true;
    signOutBtn.hidden = false;
  } else {
    authStatus.textContent = "Not signed in";
    if (authGrid) authGrid.hidden = false;
    signUpBtn.hidden = false;
    signInBtn.hidden = false;
    signOutBtn.hidden = true;
  }
}

async function loadSession() {
  if (!supabaseClient) return;
  const { data } = await supabaseClient.auth.getSession();
  currentUser = data.session?.user || null;
  updateAuthUI();
}

signUpBtn.addEventListener("click", async () => {
  if (!supabaseClient) return;
  const email = authEmailInput.value.trim();
  const password = authPasswordInput.value.trim();
  if (!email || !password) return;
  const { error } = await supabaseClient.auth.signUp({ email, password });
  if (error) {
    authStatus.textContent = `Sign up failed: ${error.message}`;
  } else {
    authStatus.textContent = "Check your email to confirm sign up.";
  }
});

signInBtn.addEventListener("click", async () => {
  if (!supabaseClient) return;
  const email = authEmailInput.value.trim();
  const password = authPasswordInput.value.trim();
  if (!email || !password) return;
  const { error } = await supabaseClient.auth.signInWithPassword({ email, password });
  if (error) {
    authStatus.textContent = `Sign in failed: ${error.message}`;
  } else {
    window.location.href = "index.html#tracker";
  }
});

signOutBtn.addEventListener("click", async () => {
  if (!supabaseClient) return;
  await supabaseClient.auth.signOut();
  currentUser = null;
  updateAuthUI();
});

if (supabaseClient) {
  supabaseClient.auth.onAuthStateChange((_event, session) => {
    currentUser = session?.user || null;
    updateAuthUI();
  });
}

loadSession();
