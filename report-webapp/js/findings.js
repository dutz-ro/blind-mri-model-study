// Findings page: renders every finding the model produced, with its evidence.

import { t } from "./i18n.js";
import {
  clinicalValue,
  fieldLabel,
  findingTitle,
  flattenClinical,
  plainSummary,
  seriesRoleLabel,
} from "./clinical.js";
import { badge, element, modelLabel, startPage } from "./page.js";
import { vascularLine } from "./vascular.js";

let dialogState = { evidence: [], index: 0, zoom: 1, dictionary: null };

function modelText(dictionary, value) {
  const wrap = element("span", "model-text");
  wrap.append(element("span", "model-text-value", value));
  wrap.append(element("span", "model-text-badge", t(dictionary, "index.modelTextBadge")));
  return wrap;
}

function clinicalTable(dictionary, clinical) {
  const table = element("dl", "clinical-table");
  for (const row of flattenClinical(clinical)) {
    if (row.group) {
      const heading = element("dt", "clinical-group", fieldLabel(dictionary, row.key));
      const spacer = element("dd", "clinical-group-value", "");
      table.append(heading, spacer);
      continue;
    }
    const term = element("dt", row.depth ? "clinical-child" : null, fieldLabel(dictionary, row.key));
    const definition = element("dd", row.depth ? "clinical-child" : null);
    if (row.freeText) {
      const list = element("ul", "free-text-list");
      for (const item of row.value) {
        const entry = element("li");
        entry.append(modelText(dictionary, item));
        list.append(entry);
      }
      definition.append(list);
    } else {
      definition.textContent = clinicalValue(dictionary, row.value);
    }
    table.append(term, definition);
  }
  return table;
}

function evidenceButton(dictionary, finding, onOpen) {
  const button = element("button", "evidence-button evidence-trigger");
  button.type = "button";
  button.append(element("span", null, t(dictionary, "index.evidenceButton")));
  button.append(element("small", null, t(dictionary, "index.evidenceButtonHint")));
  button.disabled = !finding.evidence.length;
  button.addEventListener("click", () => onOpen(finding));
  return button;
}

function findingCard(dictionary, finding, index, onOpen) {
  const article = element("article", "model-report finding-card");

  const kicker = element("div", "report-kicker");
  kicker.append(element("span", "model-index", String(index + 1).padStart(2, "0")));
  kicker.append(element("span", null, finding.findingId || ""));
  article.append(kicker);

  if (finding.provenance && finding.provenance.historical) {
    article.classList.add("historical-finding");
    article.append(
      element("p", "historical-warning", t(dictionary, "models.historicalFinding")),
    );
  }

  const heading = element("div", "model-heading");
  const titleWrap = element("div");
  titleWrap.append(element("h3", null, findingTitle(dictionary, finding.clinical)));
  titleWrap.append(
    element(
      "p",
      "eyebrow",
      `${fieldLabel(dictionary, "tear_status")}: ${clinicalValue(dictionary, finding.clinical.tear_status)}`,
    ),
  );
  heading.append(titleWrap);
  const chips = element("div", "finding-chips");
  chips.append(
    element(
      "span",
      `confidence ${finding.confidence || "low"}`,
      `${fieldLabel(dictionary, "confidence")}: ${clinicalValue(dictionary, finding.confidence)}`,
    ),
  );
  chips.append(badge(dictionary, finding.evidenceStatus));
  heading.append(chips);
  article.append(heading);

  const plain = element("div", "plain-view");
  plain.append(element("p", "conclusion-lead", plainSummary(dictionary, finding.clinical)));
  article.append(plain);

  const clinicalView = element("div", "clinical-view");
  clinicalView.hidden = true;
  clinicalView.append(clinicalTable(dictionary, finding.clinical));
  article.append(clinicalView);

  article.append(vascularLine(dictionary, finding.clinical));

  if (finding.limitations && finding.limitations.length) {
    const limits = element("div", "finding-limitations");
    limits.append(element("h4", null, t(dictionary, "index.limitationsHeading")));
    const list = element("ul", "free-text-list");
    for (const item of finding.limitations) {
      const entry = element("li");
      entry.append(modelText(dictionary, item));
      list.append(entry);
    }
    limits.append(list);
    article.append(limits);
  }

  const count = element(
    "p",
    "evidence-count",
    finding.evidence.length
      ? t(dictionary, "report.results.evidence", { count: finding.evidence.length })
      : t(dictionary, "report.results.evidenceNone"),
  );
  article.append(count);
  article.append(evidenceButton(dictionary, finding, onOpen));
  return article;
}

function renderDialogImage() {
  const { evidence, index, zoom, dictionary } = dialogState;
  const entry = evidence[index];
  const image = document.getElementById("main-evidence");
  const meta = document.getElementById("evidence-meta");
  const zoomLabel = document.getElementById("zoom-label");
  meta.replaceChildren();
  if (!entry) {
    meta.append(element("p", null, t(dictionary, "dialog.noEvidence")));
    image.removeAttribute("src");
    return;
  }
  image.src = `evidence/${entry.image}`;
  image.alt = t(dictionary, "dialog.imageAlt");
  image.style.setProperty("--zoom", String(zoom));
  zoomLabel.textContent = `${Math.round(zoom * 100)}%`;

  const list = element("dl", "evidence-meta-list");
  const rows = [
    ["dialog.meta.exportId", entry.export_id],
    ["dialog.meta.role", clinicalValue(dictionary, entry.role)],
    ["dialog.meta.series", seriesRoleLabel(dictionary, entry.seriesRole)],
    ["dialog.meta.index", entry.geometricIndex],
    ["dialog.meta.windows", (entry.windowIds || []).join(", ")],
  ];
  for (const [key, value] of rows) {
    list.append(element("dt", null, t(dictionary, key)));
    list.append(element("dd", null, value === null || value === undefined ? "—" : String(value)));
  }
  meta.append(list);
  if (entry.description) {
    const description = element("div", "evidence-description");
    description.append(element("h4", null, t(dictionary, "dialog.meta.description")));
    const paragraph = element("p");
    paragraph.append(modelText(dictionary, entry.description));
    description.append(paragraph);
    meta.append(description);
  }

  const thumbs = document.getElementById("evidence-thumbs");
  thumbs.replaceChildren();
  evidence.forEach((item, position) => {
    const thumb = element("button", position === index ? "thumb active" : "thumb");
    thumb.type = "button";
    const picture = document.createElement("img");
    picture.src = `evidence/${item.image}`;
    picture.alt = item.export_id;
    picture.loading = "lazy";
    thumb.append(picture);
    thumb.addEventListener("click", () => {
      dialogState.index = position;
      dialogState.zoom = 1;
      renderDialogImage();
    });
    thumbs.append(thumb);
  });
}

function openDialog(dictionary, finding, model) {
  const dialog = document.getElementById("evidence-dialog");
  dialogState = { evidence: finding.evidence, index: 0, zoom: 1, dictionary };
  document.getElementById("dialog-model").textContent = `${modelLabel(dictionary, model)} · ${t(dictionary, "dialog.eyebrow")}`;
  document.getElementById("dialog-title").textContent = findingTitle(dictionary, finding.clinical);
  renderDialogImage();
  if (typeof dialog.showModal === "function") {
    dialog.showModal();
  } else {
    dialog.setAttribute("open", "open");
  }
}

function wireDialogControls() {
  const dialog = document.getElementById("evidence-dialog");
  dialog.querySelector(".close-button").addEventListener("click", () => dialog.close());
  dialog.querySelectorAll("[data-zoom]").forEach((button) => {
    button.addEventListener("click", () => {
      const step = button.dataset.zoom === "in" ? 0.25 : -0.25;
      dialogState.zoom = Math.min(3, Math.max(0.5, dialogState.zoom + step));
      renderDialogImage();
    });
  });
}

function wireViewToggle() {
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => {
      const clinical = button.dataset.view === "clinical";
      document.querySelectorAll("[data-view]").forEach((item) => {
        item.classList.toggle("active", item === button);
      });
      document.querySelectorAll(".plain-view").forEach((item) => {
        item.hidden = clinical;
      });
      document.querySelectorAll(".clinical-view").forEach((item) => {
        item.hidden = !clinical;
      });
    });
  });
}

function renderCoverage(bundle) {
  const study = bundle.study || {};
  const coverage = study.coverage || {};
  document.getElementById("coverage-number").textContent =
    coverage.analyzed_unique_slices === undefined ? "—" : String(coverage.analyzed_unique_slices);
  document.getElementById("coverage-percent").textContent =
    coverage.coverage_percent === undefined ? "—" : `${coverage.coverage_percent}%`;
  document.getElementById("coverage-windows").textContent =
    coverage.windows === undefined ? "—" : String(coverage.windows);
}

function globalLimitations(dictionary, model) {
  const host = element("div", "global-limitations");
  if (!model.globalLimitations || !model.globalLimitations.length) {
    return host;
  }
  host.append(element("h3", null, t(dictionary, "index.globalLimitationsHeading")));
  const list = element("ul", "free-text-list");
  for (const item of model.globalLimitations) {
    const entry = element("li");
    entry.append(modelText(dictionary, item));
    list.append(entry);
  }
  host.append(list);
  return host;
}

function modelSource(dictionary, model) {
  const provenance = model.provenance || {};
  const key = provenance.sourceType === "dicom_windows"
    ? "models.source.dicom"
    : "models.source.generalist";
  return t(dictionary, key, {
    count: provenance.imageCount ?? "—",
    groups: provenance.windowCount ?? provenance.batchCount ?? "—",
  });
}

function modelSummary(dictionary, model) {
  const positive = model.findings.filter((finding) =>
    ["suspected", "probable"].includes(finding.clinical.tear_status));
  const historical = positive.filter((finding) => finding.provenance?.historical);
  const verified = positive.filter((finding) => !finding.provenance?.historical);
  const card = element("article", "model-overview-card");
  card.append(element("h3", null, modelLabel(dictionary, model)));
  card.append(element("p", "model-source", modelSource(dictionary, model)));
  card.append(
    element(
      "strong",
      positive.length ? "model-positive" : "model-negative",
      t(dictionary, "models.positiveCount", { count: positive.length }),
    ),
  );
  if (historical.length) {
    card.append(
      element(
        "p",
        "historical-summary",
        t(dictionary, "models.historicalCount", {
          count: historical.length,
          final: verified.length,
        }),
      ),
    );
  }
  return card;
}

function modelSection(dictionary, model) {
  const section = element("section", "model-arm-section");
  const head = element("div", "model-arm-heading");
  const title = element("div");
  title.append(element("p", "eyebrow", t(dictionary, "models.independentArm")));
  title.append(element("h2", null, modelLabel(dictionary, model)));
  title.append(element("p", "model-source", modelSource(dictionary, model)));
  head.append(title);
  head.append(
    element(
      "span",
      "finding-total",
      t(dictionary, "models.findingCount", { count: model.findings.length }),
    ),
  );
  section.append(head);
  if (!model.findings.length) {
    section.append(element("p", "empty-state", t(dictionary, "index.noFindings")));
  }
  model.findings.forEach((finding, index) => {
    section.append(
      findingCard(dictionary, finding, index, (selected) =>
        openDialog(dictionary, selected, model)),
    );
  });
  section.append(globalLimitations(dictionary, model));
  return section;
}

function render({ bundle, dictionary }) {
  const models = bundle.models || [];
  const host = document.getElementById("findings-list");
  host.replaceChildren();
  renderCoverage(bundle);

  const overview = document.getElementById("model-overview");
  overview.replaceChildren();
  const overviewGrid = element("div", "model-overview-grid");
  models.forEach((model) => overviewGrid.append(modelSummary(dictionary, model)));
  overview.append(overviewGrid);

  if (!models.length) {
    host.append(element("p", "empty-state", t(dictionary, "index.noFindings")));
    return;
  }
  models.forEach((model) => host.append(modelSection(dictionary, model)));
  wireViewToggle();
}

wireDialogControls();
startPage(render);
