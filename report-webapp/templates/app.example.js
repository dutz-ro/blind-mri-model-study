// Add study-specific conclusions and evidence IDs only in the ignored local app.js file.
document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    const clinical = button.dataset.view === "clinical";
    document.querySelectorAll("[data-view]").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    document.querySelectorAll(".plain-view").forEach((item) => { item.hidden = clinical; });
    document.querySelectorAll(".clinical-view").forEach((item) => { item.hidden = !clinical; });
  });
});

// Expected local evidence shape:
// { model, role, roleLabel, seen, supports, limitation, point: [x, y] }
// Keep identifiers and study-specific descriptions outside Git.
