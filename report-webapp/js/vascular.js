// Vascular zone reporting.
//
// The schema records a conventional zone, the basis for the estimate, a
// confidence, and an explicit statement that vascularity was never directly
// observed. The viewer keeps those four together everywhere the zone appears,
// so the estimate can never be read as a perfusion measurement, and it repeats
// that the zone decides neither repairability nor treatment.

import { t } from "./i18n.js";
import { clinicalValue } from "./clinical.js";
import { element } from "./page.js";

export function zoneOf(finding) {
  const clinical = finding.clinical || finding;
  return (clinical.vascular_zone_estimate || {}).zone || "indeterminate";
}

export function zoneDistribution(findings) {
  const counts = new Map();
  for (const finding of findings) {
    const zone = zoneOf(finding);
    counts.set(zone, (counts.get(zone) || 0) + 1);
  }
  return [...counts.entries()].sort((left, right) => right[1] - left[1]);
}

export function zoneSummarySentence(dictionary, findings) {
  const distribution = zoneDistribution(findings);
  if (!distribution.length) {
    return "";
  }
  if (distribution.length === 1) {
    return t(dictionary, "vascular.summary.allSame", {
      total: findings.length,
      zone: clinicalValue(dictionary, distribution[0][0]),
    });
  }
  return t(dictionary, "vascular.summary.mixed");
}

// The per-finding line shown inside a finding card.
export function vascularLine(dictionary, clinical) {
  const estimate = clinical.vascular_zone_estimate || {};
  const wrap = element("div", "vascular-line");
  wrap.append(element("h4", null, t(dictionary, "vascular.heading")));
  const list = element("dl", "vascular-facts");
  const rows = [
    ["vascular.zoneLabel", clinicalValue(dictionary, estimate.zone)],
    ["vascular.basisLabel", clinicalValue(dictionary, estimate.basis)],
    ["vascular.confidenceLabel", clinicalValue(dictionary, estimate.confidence)],
  ];
  for (const [key, value] of rows) {
    const term = element("dt", null, t(dictionary, key));
    const definition = element("dd", null, value);
    list.append(term, definition);
  }
  wrap.append(list);
  wrap.append(
    element("p", "vascular-interpretation", t(dictionary, `vascular.interpretation.${estimate.zone || "indeterminate"}`)),
  );
  wrap.append(element("p", "vascular-guard", t(dictionary, "vascular.notObserved")));
  wrap.append(element("p", "vascular-guard", t(dictionary, "vascular.notRepairability")));
  return wrap;
}

// The aggregate panel shown once per page.
export function vascularPanel(dictionary, findings) {
  const section = element("section", "vascular-panel");
  section.append(element("h2", null, t(dictionary, "vascular.panelHeading")));
  section.append(element("p", "panel-lede", t(dictionary, "vascular.panelLede")));
  section.append(element("p", "vascular-summary", zoneSummarySentence(dictionary, findings)));

  const distribution = zoneDistribution(findings);
  const list = element("ul", "zone-distribution");
  list.setAttribute("aria-label", t(dictionary, "vascular.distribution"));
  for (const [zone, count] of distribution) {
    const item = element("li", `zone-chip zone-${zone}`);
    item.append(element("span", "zone-name", clinicalValue(dictionary, zone)));
    item.append(element("span", "zone-count", `${count}/${findings.length}`));
    item.append(
      element("span", "zone-meaning", t(dictionary, `vascular.interpretation.${zone}`)),
    );
    list.append(item);
  }
  section.append(list);
  section.append(element("p", "vascular-guard", t(dictionary, "vascular.notObserved")));
  section.append(element("p", "vascular-guard", t(dictionary, "vascular.notRepairability")));
  return section;
}
