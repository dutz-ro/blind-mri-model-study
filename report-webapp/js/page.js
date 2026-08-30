// Shared page bootstrap: language, chrome, and study data.
//
// Every page loads the same bundle and the same dictionaries, then re-renders
// in place when the language changes. Nothing is fetched from outside the
// origin, so the strict content security policy stays intact.

import {
  applyStaticTranslations,
  loadDictionary,
  mountLanguageSelector,
  preferredLanguage,
  rememberLanguage,
  t,
} from "./i18n.js";

export const BUNDLE_URL = "report-bundle.json";

export async function loadBundle() {
  const response = await fetch(BUNDLE_URL, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Missing bundle: ${response.status}`);
  }
  return response.json();
}

export function primaryModel(bundle) {
  const models = bundle.models || [];
  return models.length ? models[0] : null;
}

export function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text !== undefined && text !== null) {
    node.textContent = String(text);
  }
  return node;
}

export function formatTimestamp(value, language) {
  if (!value) {
    return "";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return String(value);
  }
  return parsed.toLocaleString(language, { dateStyle: "medium", timeStyle: "short" });
}

// A finding without usable image evidence is shown and labelled, never hidden.
export function badge(dictionary, status) {
  const node = element("span", `evidence-status status-${status}`);
  node.textContent = t(dictionary, `clinical.status.${status}`);
  return node;
}

export async function startPage(render) {
  const root = document.body;
  let language = preferredLanguage();
  let dictionary = await loadDictionary(language);
  let bundle = null;
  let loadError = null;
  try {
    bundle = await loadBundle();
  } catch (error) {
    loadError = error;
  }

  const selectorHost = root.querySelector("[data-language-selector]");

  async function apply(code) {
    language = code;
    dictionary = await loadDictionary(code);
    rememberLanguage(code);
    document.documentElement.lang = dictionary.meta.htmlLang;
    document.documentElement.dir = dictionary.meta.dir || "ltr";
    applyStaticTranslations(dictionary);
    if (selectorHost) {
      const { label } = mountLanguageSelector(selectorHost, code, (next) => {
        apply(next);
      });
      label.textContent = t(dictionary, "app.language.label");
      label.setAttribute("aria-label", t(dictionary, "app.language.aria"));
    }
    const status = root.querySelector("[data-load-status]");
    if (status) {
      status.textContent = loadError ? t(dictionary, "app.loadError") : "";
      status.hidden = !loadError;
    }
    if (!loadError) {
      render({ bundle, dictionary, language });
    }
  }

  await apply(language);
}
