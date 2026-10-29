"use strict";

/** Browser state is kept separate from the ranker's scores and model artifacts. */
const elements = Object.fromEntries(
  ["search", "matches", "selected", "mode", "relevance-weight", "recommend", "model-note", "error", "recommendations"]
    .map(id => [id, document.getElementById(id)])
);
const selected = new Map();
let searchTimer;
let searchRevision = 0;
let rankingRevision = 0;
let searchRequest;
let rankingRequest;
let rankingBusy = false;

async function readJson(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}

function updateButton() {
  elements.recommend.disabled = rankingBusy || !selected.size;
  elements.recommend.textContent = rankingBusy ? "Finding recommendations…" : "Find recommendations";
  elements.recommendations.setAttribute("aria-busy", String(rankingBusy));
}

/** Reject a late result if the user has changed their history or ranking mode. */
function invalidateRanking() {
  rankingRevision += 1;
  rankingRequest?.abort();
  rankingBusy = false;
  updateButton();
  elements.recommendations.textContent = selected.size
    ? "Selection changed. Request recommendations for this history."
    : "Recommendations will appear here after you choose a film.";
  elements.recommendations.removeAttribute("data-mode");
  elements.recommendations.removeAttribute("data-relevance-weight");
  elements["model-note"].textContent = "No current ranking. Request recommendations for the selected history and approach.";
  elements.error.textContent = "";
}

function renderSelected() {
  elements.selected.replaceChildren();
  for (const [identifier, title] of selected) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.append(document.createTextNode(title));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${title}`);
    remove.addEventListener("click", () => {
      selected.delete(identifier);
      renderSelected();
      invalidateRanking();
    });
    chip.append(remove);
    elements.selected.append(chip);
  }
  if (!selected.size) elements.selected.textContent = "No films selected.";
  updateButton();
}

function addMovie(movie) {
  if (selected.has(movie.movie_id)) return;
  if (selected.size >= 20) {
    elements.error.textContent = "Choose up to 20 liked films; remove one before adding another.";
    return;
  }
  selected.set(movie.movie_id, movie.title);
  renderSelected();
  invalidateRanking();
}

function renderMatches(movies) {
  elements.matches.replaceChildren();
  for (const movie of movies) {
    const row = document.createElement("div");
    row.className = "result";
    row.append(document.createTextNode(movie.title));
    const add = document.createElement("button");
    add.type = "button";
    add.textContent = "+";
    add.setAttribute("aria-label", `Add ${movie.title}`);
    add.addEventListener("click", () => addMovie(movie));
    row.append(add);
    elements.matches.append(row);
  }
  if (!movies.length) elements.matches.textContent = "No matching films in this catalog.";
}

elements.search.addEventListener("input", event => {
  clearTimeout(searchTimer);
  searchRequest?.abort();
  const current = ++searchRevision;
  const query = event.target.value.trim();
  elements.matches.replaceChildren();
  if (query.length < 2) return;
  // Debouncing limits local requests; the revision also guards cached/late responses.
  searchTimer = setTimeout(async () => {
    searchRequest = new AbortController();
    try {
      const movies = await fetch(`/movies?q=${encodeURIComponent(query)}`, {
        signal: searchRequest.signal,
      }).then(readJson);
      if (current === searchRevision) renderMatches(movies);
    } catch (error) {
      if (error.name !== "AbortError" && current === searchRevision) elements.error.textContent = error.message;
    }
  }, 250);
});

elements.mode.addEventListener("change", invalidateRanking);
elements["relevance-weight"].addEventListener("change", invalidateRanking);
elements.recommend.addEventListener("click", async () => {
  const current = ++rankingRevision;
  const mode = elements.mode.value;
  const relevanceWeight = Number(elements["relevance-weight"].value);
  rankingRequest?.abort();
  rankingRequest = new AbortController();
  rankingBusy = true;
  updateButton();
  elements.error.textContent = "";
  // A new attempt invalidates displayed results even when the inputs are unchanged.
  elements.recommendations.textContent = "Loading recommendations…";
  elements.recommendations.removeAttribute("data-mode");
  elements.recommendations.removeAttribute("data-relevance-weight");
  elements["model-note"].textContent = "Loading the selected ranking…";
  try {
    const result = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: rankingRequest.signal,
      body: JSON.stringify({ movie_ids: [...selected.keys()], k: 10, mode, relevance_weight: relevanceWeight }),
    }).then(readJson);
    if (current !== rankingRevision) return;
    elements.recommendations.replaceChildren();
    for (const [index, movie] of result.recommendations.entries()) {
      const row = document.createElement("div");
      row.className = "result";
      const text = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = `${index + 1}. ${movie.title}`;
      const genres = document.createElement("div");
      genres.className = "small";
      genres.textContent = movie.genres.join(" · ");
      text.append(title, genres);
      row.append(text);
      elements.recommendations.append(row);
    }
    elements.recommendations.dataset.mode = mode;
    elements.recommendations.dataset.relevanceWeight = String(relevanceWeight);
    const bucket = result.selector_history_bucket ? ` History bucket: ${result.selector_history_bucket}.` : "";
    const variety = result.reranking?.enabled
      ? ` Genre variety reranking: ${Math.round(result.reranking.relevance_weight * 100)}% relevance weight within the top ${result.reranking.candidate_pool} candidates. Returned scores remain original relevance scores.`
      : " Original relevance order; genre variety is off.";
    elements["model-note"].textContent = `Showing ${result.selected_ranker} ranking.${bucket}${variety} Scores are not probabilities.`;
  } catch (error) {
    if (error.name !== "AbortError" && current === rankingRevision) {
      elements.error.textContent = error.message;
      elements.recommendations.textContent = "Recommendations are unavailable for the current request.";
      elements["model-note"].textContent = "No current ranking. Retry the request when the service is available.";
    }
  } finally {
    if (current === rankingRevision) {
      rankingBusy = false;
      updateButton();
    }
  }
});

async function loadModelInfo() {
  try {
    const info = await fetch("/model-info").then(readJson);
    // Earlier bundles remain usable, but must not advertise unavailable rankers.
    for (const option of elements.mode.options) {
      if (info.format_version === 1) option.disabled = !["accuracy", "neural"].includes(option.value);
      if (info.format_version === 2) option.disabled = ["adaptive", "als", "blend"].includes(option.value);
    }
    if (info.history_selector) {
      const policy = Object.entries(info.history_selector)
        .map(([count, method]) => `${count} likes → ${typeof method === "string" ? method : method.name}`)
        .join("; ");
      if (rankingRevision === 0) elements["model-note"].textContent = `Validation-selected policy: ${policy}.`;
    } else {
      if (rankingRevision !== 0) return;
      elements["model-note"].textContent = info.format_version === 2
        ? `Accuracy model: ${info.accuracy_model}. Hybrid includes ${info.seeds.length} independently trained neural models.`
        : "Original neural checkpoint. Run the documented experiment to enable additional approaches.";
    }
  } catch (error) {
    if (rankingRevision === 0) elements["model-note"].textContent = `Model details unavailable: ${error.message}`;
  }
}

loadModelInfo();
