// Translate the study's structured vocabulary.
//
// Every clinical value in the bundle is a schema enum, so it can be translated
// exactly. Anything that is not an enum is model free text and is shown
// verbatim with a language badge, because translating it would mean inventing
// clinical wording the model never produced.

import { t } from "./i18n.js";

export function clinicalValue(dictionary, value) {
  if (value === null || value === undefined || value === "") {
    return t(dictionary, "clinical.value.indeterminate");
  }
  if (typeof value === "boolean") {
    return t(dictionary, `clinical.value.${value ? "true" : "false"}`);
  }
  if (typeof value === "number") {
    return String(value);
  }
  const translated = t(dictionary, `clinical.value.${value}`);
  return translated === `clinical.value.${value}` ? String(value) : translated;
}

export function fieldLabel(dictionary, key) {
  const translated = t(dictionary, `clinical.field.${key}`);
  return translated === `clinical.field.${key}` ? key : translated;
}

export function statusLabel(dictionary, status) {
  const translated = t(dictionary, `clinical.status.${status}`);
  return translated === `clinical.status.${status}` ? status : translated;
}

export function seriesRoleLabel(dictionary, role) {
  const translated = t(dictionary, `clinical.role.${role}`);
  return translated === `clinical.role.${role}` ? role : translated;
}

// The bundle keeps clinical fields in a nested object; the viewer shows them as
// flat, labelled rows so a reader can compare findings without unfolding trees.
export function flattenClinical(clinical) {
  const rows = [];
  for (const [key, value] of Object.entries(clinical || {})) {
    if (value === null || value === undefined) {
      rows.push({ key, value: null, depth: 0 });
    } else if (Array.isArray(value)) {
      if (value.length) {
        rows.push({ key, value, depth: 0, freeText: true });
      }
    } else if (typeof value === "object") {
      rows.push({ key, group: true, depth: 0 });
      for (const [childKey, childValue] of Object.entries(value)) {
        if (Array.isArray(childValue)) {
          if (childValue.length) {
            rows.push({ key: childKey, value: childValue, depth: 1, freeText: true });
          }
        } else {
          rows.push({ key: childKey, value: childValue, depth: 1 });
        }
      }
    } else {
      rows.push({ key, value, depth: 0 });
    }
  }
  return rows;
}

export function findingTitle(dictionary, clinical) {
  const meniscus = clinicalValue(dictionary, clinical.meniscus);
  const segment = clinicalValue(dictionary, clinical.segment);
  return `${meniscus} · ${segment}`;
}

// A one-sentence restatement of the structured fields, never of model prose.
export function plainSummary(dictionary, clinical) {
  const parts = [];
  const status = clinical.tear_status;
  const tearKey =
    {
      absent: "report.results.tearAbsent",
      suspected: "report.results.tearSuspected",
      probable: "report.results.tearProbable",
      indeterminate: "report.results.tearIndeterminate",
      not_evaluable: "report.results.tearNotEvaluable",
    }[status] || "report.results.tearIndeterminate";
  parts.push(t(dictionary, tearKey));
  if (clinical.tear_pattern && clinical.tear_pattern !== "none") {
    parts.push(
      t(dictionary, "report.results.pattern", {
        pattern: clinicalValue(dictionary, clinical.tear_pattern),
      }),
    );
  }
  const zone = (clinical.vascular_zone_estimate || {}).zone;
  if (zone) {
    parts.push(
      t(dictionary, "report.results.vascular", {
        zone: clinicalValue(dictionary, zone),
        basis: clinicalValue(dictionary, (clinical.vascular_zone_estimate || {}).basis),
      }),
    );
  }
  const sentence = parts.join(", ");
  return sentence.charAt(0).toUpperCase() + sentence.slice(1) + ".";
}
