// Radiology-style report page.
//
// The narrative is composed from the study's structured fields only. No model
// prose is reused, so the same report can be produced in three languages from
// one bundle and every sentence can be traced back to a schema value.
//
// The study analysed the menisci and nothing else. Structures the acquisition
// contains but the study never examined are listed explicitly as not assessed,
// because silence in a radiology report reads as normality and would be a claim
// this study cannot support.

import { t } from "./i18n.js";
import { clinicalValue, seriesRoleLabel } from "./clinical.js";
import { element, formatTimestamp, primaryModel, startPage } from "./page.js";
import { vascularPanel, zoneSummarySentence } from "./vascular.js";

const TEAR_PHRASES = {
  absent: "report.results.tearAbsent",
  suspected: "report.results.tearSuspected",
  probable: "report.results.tearProbable",
  indeterminate: "report.results.tearIndeterminate",
  not_evaluable: "report.results.tearNotEvaluable",
};

const MENISCUS_ORDER = ["medial", "lateral", "indeterminate"];
const MENISCUS_HEADINGS = {
  medial: "report.results.medial",
  lateral: "report.results.lateral",
  indeterminate: "report.results.indeterminateMeniscus",
};

function allAbsent(group, keys) {
  return keys.every((key) => (group || {})[key] === "absent");
}

function findingSentence(dictionary, finding) {
  const clinical = finding.clinical;
  const parts = [];

  parts.push(t(dictionary, TEAR_PHRASES[clinical.tear_status] || TEAR_PHRASES.indeterminate));

  if (clinical.tear_pattern && clinical.tear_pattern !== "none") {
    parts.push(
      t(dictionary, "report.results.pattern", {
        pattern: clinicalValue(dictionary, clinical.tear_pattern),
      }),
    );
  }
  if (clinical.orientation && !["not_applicable", "indeterminate"].includes(clinical.orientation)) {
    parts.push(
      t(dictionary, "report.results.orientation", {
        orientation: clinicalValue(dictionary, clinical.orientation),
      }),
    );
  }

  const surface = clinical.surface_extension || {};
  if (allAbsent(surface, ["superior", "inferior", "free_edge"])) {
    parts.push(t(dictionary, "report.results.surfaceIntact"));
  } else {
    parts.push(
      t(dictionary, "report.results.surfaceDetail", {
        superior: clinicalValue(dictionary, surface.superior),
        inferior: clinicalValue(dictionary, surface.inferior),
        freeEdge: clinicalValue(dictionary, surface.free_edge),
      }),
    );
  }

  const capsular = clinical.capsular_relation || {};
  if (allAbsent(capsular, ["meniscocapsular_separation", "peripheral_extension"])) {
    parts.push(t(dictionary, "report.results.capsularIntact"));
  } else {
    parts.push(
      t(dictionary, "report.results.capsularDetail", {
        separation: clinicalValue(dictionary, capsular.meniscocapsular_separation),
        extension: clinicalValue(dictionary, capsular.peripheral_extension),
      }),
    );
  }
  // A zero-millimetre visual estimate carries no information and reads as a
  // measurement; only a positive distance is stated, with its method.
  if (
    typeof capsular.distance_mm_estimate === "number" &&
    capsular.distance_mm_estimate > 0 &&
    capsular.measurement_method &&
    capsular.measurement_method !== "not_measurable"
  ) {
    parts.push(
      t(dictionary, "report.results.distance", {
        value: capsular.distance_mm_estimate,
        method: clinicalValue(dictionary, capsular.measurement_method),
      }),
    );
  }

  const healing = clinical.healing_context || {};
  const cystKey =
    {
      absent: "report.results.cystAbsent",
      present: "report.results.cystPresent",
    }[healing.parameniscal_cyst] || "report.results.cystIndeterminate";
  parts.push(t(dictionary, cystKey));
  const rootKey =
    {
      absent: "report.results.rootAbsent",
      present: "report.results.rootPresent",
    }[healing.root_involvement] || "report.results.rootIndeterminate";
  parts.push(t(dictionary, rootKey));

  const stability = clinical.stability_features || {};
  if (allAbsent(stability, ["displaced_fragment", "flap", "bucket_handle_pattern"])) {
    parts.push(t(dictionary, "report.results.fragmentAbsent"));
  } else {
    parts.push(
      t(dictionary, "report.results.fragmentDetail", {
        displaced: clinicalValue(dictionary, stability.displaced_fragment),
        flap: clinicalValue(dictionary, stability.flap),
        bucket: clinicalValue(dictionary, stability.bucket_handle_pattern),
      }),
    );
  }

  parts.push(
    t(dictionary, "report.results.tissue", {
      tissue: clinicalValue(dictionary, healing.tissue_quality_imaging_estimate),
      chronicity: clinicalValue(dictionary, healing.chronicity_imaging_estimate),
    }),
  );

  const vascular = clinical.vascular_zone_estimate || {};
  parts.push(
    t(dictionary, "report.results.vascular", {
      zone: clinicalValue(dictionary, vascular.zone),
      basis: clinicalValue(dictionary, vascular.basis),
    }),
  );

  parts.push(
    finding.evidence.length
      ? t(dictionary, "report.results.evidence", { count: finding.evidence.length })
      : t(dictionary, "report.results.evidenceNone"),
  );
  parts.push(
    t(dictionary, "report.results.confidence", {
      confidence: clinicalValue(dictionary, finding.confidence),
    }),
  );

  return `${parts.join(", ")}.`;
}

function renderHeader(dictionary, bundle, language) {
  const study = bundle.study || {};
  const coverage = study.coverage || {};
  const host = document.getElementById("report-header");
  host.replaceChildren();
  host.append(element("h2", null, t(dictionary, "report.header.heading")));
  const list = element("dl", "report-header-list");
  const sequences = (study.sequences || [])
    .map(
      (item) =>
        `${item.seriesDescription} (${seriesRoleLabel(dictionary, item.role)}, ${item.windows})`,
    )
    .join(" · ");
  const rows = [
    [t(dictionary, "report.header.examination"), t(dictionary, "report.header.examinationValue")],
    [
      t(dictionary, "report.header.side"),
      `${clinicalValue(dictionary, study.laterality)} (${t(dictionary, "report.header.sideNote")})`,
    ],
    [t(dictionary, "report.header.sequences"), sequences || "—"],
    [
      t(dictionary, "report.header.coverage"),
      t(dictionary, "report.header.coverageValue", {
        analyzed: coverage.analyzed_unique_slices ?? "—",
        total: coverage.total_source_slices ?? "—",
        percent: coverage.coverage_percent ?? "—",
        windows: coverage.windows ?? "—",
      }),
    ],
    [t(dictionary, "report.header.purpose"), t(dictionary, "report.header.purposeValue")],
    [t(dictionary, "report.header.priorReport"), t(dictionary, "report.header.priorReportValue")],
    [t(dictionary, "report.header.run"), study.run || "—"],
  ];
  for (const [term, value] of rows) {
    list.append(element("dt", null, term));
    list.append(element("dd", null, value));
  }
  host.append(list);
  const footer = document.getElementById("report-footnote");
  footer.textContent = t(dictionary, "report.footerNote", {
    timestamp: formatTimestamp(bundle.generated_at, language),
    run: study.run || "—",
  });
}

function renderResults(dictionary, model) {
  const host = document.getElementById("report-results");
  host.replaceChildren();
  host.append(element("h2", null, t(dictionary, "report.results.heading")));

  const grouped = new Map();
  for (const finding of model.findings) {
    const meniscus = finding.clinical.meniscus || "indeterminate";
    if (!grouped.has(meniscus)) {
      grouped.set(meniscus, []);
    }
    grouped.get(meniscus).push(finding);
  }

  for (const meniscus of MENISCUS_ORDER) {
    const findings = grouped.get(meniscus);
    const block = element("section", "report-block");
    block.append(element("h3", null, t(dictionary, MENISCUS_HEADINGS[meniscus])));
    if (!findings || !findings.length) {
      block.append(element("p", null, t(dictionary, "report.results.noFindingsFor")));
      host.append(block);
      continue;
    }
    for (const finding of findings) {
      const paragraph = element("p", "report-sentence");
      paragraph.append(
        element(
          "strong",
          "segment-label",
          t(dictionary, "report.results.segmentIntro", {
            segment: clinicalValue(dictionary, finding.clinical.segment),
          }),
        ),
      );
      paragraph.append(document.createTextNode(` ${findingSentence(dictionary, finding)}`));
      block.append(paragraph);
    }
    host.append(block);
  }
}

function renderNotAssessed(dictionary) {
  const host = document.getElementById("report-not-assessed");
  host.replaceChildren();
  host.append(element("h2", null, t(dictionary, "report.notAssessed.heading")));
  host.append(element("p", "panel-lede", t(dictionary, "report.notAssessed.lede")));
  const list = element("ul", "not-assessed-list");
  for (const item of t(dictionary, "report.notAssessed.items") || []) {
    list.append(element("li", null, item));
  }
  host.append(list);
}

function renderConclusion(dictionary, bundle, model) {
  const study = bundle.study || {};
  const host = document.getElementById("report-conclusion");
  host.replaceChildren();
  host.append(element("h2", null, t(dictionary, "report.conclusion.heading")));

  const side = clinicalValue(dictionary, study.laterality);
  const withTear = model.findings.filter((item) =>
    ["suspected", "probable"].includes(item.clinical.tear_status),
  );
  const undetermined = model.findings.filter((item) =>
    ["indeterminate", "not_evaluable"].includes(item.clinical.tear_status),
  );

  host.append(
    element(
      "p",
      null,
      withTear.length
        ? t(dictionary, "report.conclusion.someTear", { side, count: withTear.length })
        : t(dictionary, "report.conclusion.noTear", { side }),
    ),
  );
  if (undetermined.length) {
    host.append(
      element(
        "p",
        null,
        t(dictionary, "report.conclusion.indeterminate", { count: undetermined.length }),
      ),
    );
  }
  host.append(
    element(
      "p",
      null,
      t(dictionary, "report.conclusion.vascular", {
        summary: zoneSummarySentence(dictionary, model.findings),
      }),
    ),
  );
  const windowSummary = study.windowEvidenceSummary || {};
  const counts = windowSummary.counts || {};
  if (windowSummary.total_findings) {
    host.append(
      element(
        "p",
        null,
        t(dictionary, "report.conclusion.traceability", {
          verified: counts.verified ?? 0,
          total: windowSummary.total_findings,
          percent: windowSummary.evidence_coverage_percent ?? "—",
        }),
      ),
    );
  }
  host.append(element("p", "report-boundary", t(dictionary, "report.conclusion.boundary")));
}

function render({ bundle, dictionary, language }) {
  const model = primaryModel(bundle);
  renderHeader(dictionary, bundle, language);
  const panelHost = document.getElementById("report-vascular");
  panelHost.replaceChildren();
  if (model && model.findings.length) {
    panelHost.append(vascularPanel(dictionary, model.findings));
    renderResults(dictionary, model);
    renderConclusion(dictionary, bundle, model);
  }
  renderNotAssessed(dictionary);
}

document.getElementById("print-report").addEventListener("click", () => window.print());
startPage(render);
