// Language handling for the private report viewer.
//
// The pages carry no study text: every visible string is either a translation
// key resolved here or a value taken from report-bundle.json and translated
// through the clinical vocabulary. That keeps the committed application free of
// study content and lets one bundle be read in three languages.

export const SUPPORTED_LANGUAGES = ["ro", "en", "fr"];
export const DEFAULT_LANGUAGE = "ro";
const STORAGE_KEY = "atlas-irm-language";

export function preferredLanguage() {
  // An explicit ?lang= wins, so a link can be shared in a chosen language.
  const requested = new URLSearchParams(window.location.search).get("lang");
  if (requested && SUPPORTED_LANGUAGES.includes(requested)) {
    return requested;
  }
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored && SUPPORTED_LANGUAGES.includes(stored)) {
      return stored;
    }
  } catch (error) {
    // A browser with storage disabled still gets a usable page.
  }
  for (const candidate of navigator.languages || [navigator.language || ""]) {
    const code = String(candidate).slice(0, 2).toLowerCase();
    if (SUPPORTED_LANGUAGES.includes(code)) {
      return code;
    }
  }
  return DEFAULT_LANGUAGE;
}

export function rememberLanguage(code) {
  try {
    window.localStorage.setItem(STORAGE_KEY, code);
  } catch (error) {
    // Not remembering the choice is acceptable; the page still works.
  }
}

export async function loadDictionary(code) {
  const response = await fetch(`i18n/${code}.json`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Missing dictionary: ${code}`);
  }
  return response.json();
}

export function resolve(dictionary, path) {
  return path
    .split(".")
    .reduce((node, key) => (node && node[key] !== undefined ? node[key] : undefined), dictionary);
}

export function t(dictionary, path, variables) {
  const value = resolve(dictionary, path);
  if (typeof value !== "string") {
    return value === undefined ? path : value;
  }
  if (!variables) {
    return value;
  }
  return value.replace(/\{(\w+)\}/g, (match, name) =>
    variables[name] === undefined ? match : String(variables[name]),
  );
}

// Elements declare their key in markup, so the HTML shell stays free of text.
export function applyStaticTranslations(dictionary, root = document) {
  root.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(dictionary, element.dataset.i18n);
  });
  root.querySelectorAll("[data-i18n-attr]").forEach((element) => {
    for (const pair of element.dataset.i18nAttr.split(",")) {
      const [attribute, key] = pair.split(":").map((item) => item.trim());
      if (attribute && key) {
        element.setAttribute(attribute, t(dictionary, key));
      }
    }
  });
  const title = root.querySelector("title[data-i18n-title]");
  if (title) {
    document.title = t(dictionary, title.dataset.i18nTitle);
  }
}

export function mountLanguageSelector(container, current, onChange) {
  container.replaceChildren();
  const label = document.createElement("label");
  label.className = "language-label";
  label.textContent = "";
  const select = document.createElement("select");
  select.className = "language-select";
  select.id = "language-select";
  for (const code of SUPPORTED_LANGUAGES) {
    const option = document.createElement("option");
    option.value = code;
    option.textContent = code.toUpperCase();
    option.selected = code === current;
    select.append(option);
  }
  select.addEventListener("change", () => onChange(select.value));
  label.setAttribute("for", select.id);
  container.append(label, select);
  return { label, select };
}
