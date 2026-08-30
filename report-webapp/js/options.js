// Topics page: educational prompts, never a treatment recommendation.

import { t } from "./i18n.js";
import { element, startPage } from "./page.js";

function render({ dictionary }) {
  const host = document.getElementById("options-list");
  host.replaceChildren();
  for (const item of t(dictionary, "options.items") || []) {
    const card = element("article", "option-card");
    card.append(element("h2", null, item.title));
    card.append(element("p", null, item.body));
    host.append(card);
  }
}

startPage(render);
