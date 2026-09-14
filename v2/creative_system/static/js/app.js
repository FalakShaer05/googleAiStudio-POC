(function () {
  const cfg = window.CREATIVE_SYSTEM || {};

  function switchTab(tab) {
    document.querySelectorAll(".nav-tab[data-tab]").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === tab);
    });
    document.querySelectorAll(".content-section").forEach((section) => {
      section.classList.toggle("active", section.id === "section-" + tab);
    });
    const url = new URL(window.location.href);
    url.searchParams.set("station", tab);
    window.history.replaceState({}, "", url);
  }

  function selectedWords(form) {
    return Array.from(form.querySelectorAll(".word-chip:checked")).map((el) => el.value);
  }

  function fileCache(form) {
    if (!form._fileCache) form._fileCache = {};
    return form._fileCache;
  }

  function rememberFileInput(form, input) {
    fileCache(form)[input.name] = input.files && input.files.length
      ? Array.from(input.files)
      : [];
  }

  function restoreCachedFiles(form) {
    const cache = fileCache(form);
    form.querySelectorAll('input[type="file"]').forEach((input) => {
      if (input.files && input.files.length) {
        rememberFileInput(form, input);
        return;
      }
      const files = cache[input.name];
      if (!files || !files.length || typeof DataTransfer === "undefined") return;
      const transfer = new DataTransfer();
      files.forEach((file) => transfer.items.add(file));
      input.files = transfer.files;
    });
  }

  function buildFormData(form, stationId) {
    restoreCachedFiles(form);
    const body = new FormData(form);
    body.set("station", stationId);
    body.set("_regen", String(Date.now()));
    if (form.querySelector(".word-chip")) {
      body.set("words", JSON.stringify(selectedWords(form)));
    }
    Object.entries(fileCache(form)).forEach(([name, files]) => {
      if (!files || !files.length) return;
      const current = body.getAll(name).filter((value) => value instanceof File && value.size);
      if (current.length) return;
      body.delete(name);
      files.forEach((file) => body.append(name, file, file.name));
    });
    return body;
  }

  function cacheBust(url) {
    if (!url) return url;
    return url + (url.includes("?") ? "&" : "?") + "t=" + Date.now();
  }

  function setBusy(form, submitBtn, busy) {
    form.dataset.generating = busy ? "1" : "0";
    if (!submitBtn) return;
    submitBtn.disabled = busy;
    submitBtn.setAttribute("aria-busy", busy ? "true" : "false");
    if (busy) {
      submitBtn.dataset.idleLabel = submitBtn.dataset.idleLabel || submitBtn.textContent.trim();
      submitBtn.textContent = "Generating...";
    } else if (form.dataset.hasResult === "1") {
      const stationId = form.getAttribute("data-station-form");
      submitBtn.textContent = stationId === "audio-to-text" ? "Transcribe again" : "Generate again";
    } else {
      submitBtn.textContent = submitBtn.dataset.idleLabel || "Generate";
    }
  }

  async function submitForm(form) {
    if (form.dataset.generating === "1") return;

    const stationId = form.getAttribute("data-station-form");
    const status = form.querySelector(".cs-status");
    const progress = form.querySelector(".cs-progress");
    const result = form.querySelector(".cs-result");
    const submitBtn = form.querySelector(".convert-btn");

    restoreCachedFiles(form);
    const skippedRequired = [];
    form.querySelectorAll('input[type="file"][required]').forEach((input) => {
      const cached = fileCache(form)[input.name];
      if ((!input.files || !input.files.length) && cached && cached.length) {
        input.required = false;
        skippedRequired.push(input);
      }
    });
    const valid = form.checkValidity();
    skippedRequired.forEach((input) => {
      input.required = true;
    });
    if (!valid) {
      form.reportValidity();
      return;
    }

    setBusy(form, submitBtn, true);
    if (status) status.style.display = "none";
    if (progress) {
      const note = progress.querySelector("p");
      if (note) {
        note.textContent = form.getAttribute("data-progress") || "Generating with Gemini...";
      }
      progress.style.display = "block";
    }

    try {
      const response = await fetch(cfg.generateUrl, {
        method: "POST",
        body: buildFormData(form, stationId),
        cache: "no-store",
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Generation failed");
      }
      const img = form.querySelector(".cs-result-image");
      const textEl = form.querySelector(".cs-result-text");
      const download = form.querySelector(".cs-download");
      const message = form.querySelector(".cs-result-message");
      const isText = data.result_type === "text" || (data.output_filename || "").toLowerCase().endsWith(".txt");
      if (isText) {
        if (img) {
          img.removeAttribute("src");
          img.style.display = "none";
        }
        if (textEl) {
          textEl.textContent = data.transcript || "";
          textEl.style.display = "block";
        }
      } else {
        const imageUrl = data.image_url || data.local_path || (cfg.downloadPrefix + data.output_filename);
        if (textEl) {
          textEl.textContent = "";
          textEl.style.display = "none";
        }
        if (img) {
          img.style.display = "";
          img.src = cacheBust(imageUrl);
        }
      }
      if (download) {
        download.href = cfg.downloadPrefix + data.output_filename;
        download.setAttribute("download", data.output_filename);
      }
      if (message) {
        message.textContent = data.message || (isText ? "Transcript generated successfully." : "Artwork generated successfully.");
      }
      if (result) result.style.display = "block";
      form.dataset.hasResult = "1";
      if (status) {
        status.className = "status-message status-success";
        status.textContent = data.message || "Done.";
        status.style.display = "block";
      }
    } catch (err) {
      if (status) {
        status.className = "status-message status-error";
        status.textContent = err.message || String(err);
        status.style.display = "block";
      }
    } finally {
      if (progress) progress.style.display = "none";
      setBusy(form, submitBtn, false);
    }
  }

  document.querySelectorAll(".nav-tab[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.getAttribute("data-tab")));
  });

  document.querySelectorAll("form[data-station-form]").forEach((form) => {
    form.setAttribute("novalidate", "");
    form.querySelectorAll('input[type="file"]').forEach((input) => {
      input.addEventListener("change", () => rememberFileInput(form, input));
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      submitForm(form);
    });
  });

  document.querySelectorAll("[data-filter-continue]").forEach((button) => {
    button.addEventListener("click", () => {
      const section = button.closest(".content-section");
      const guidelines = section && section.querySelector("[data-filter-guidelines]");
      const form = section && section.querySelector("[data-filter-form]");
      if (guidelines) guidelines.style.display = "none";
      if (form) {
        form.style.display = "block";
        const textarea = form.querySelector('textarea[name="text"]');
        if (textarea) textarea.focus();
      }
    });
  });

  document.querySelectorAll("[data-filter-back]").forEach((button) => {
    button.addEventListener("click", () => {
      const section = button.closest(".content-section");
      const guidelines = section && section.querySelector("[data-filter-guidelines]");
      const form = section && section.querySelector("[data-filter-form]");
      if (form) form.style.display = "none";
      if (guidelines) guidelines.style.display = "block";
    });
  });

  document.querySelectorAll("form[data-filter-form]").forEach((form) => {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!form.checkValidity()) {
        form.reportValidity();
        return;
      }

      const submitBtn = form.querySelector(".convert-btn");
      const progress = form.querySelector(".cs-progress");
      const status = form.querySelector(".cs-status");
      const decision = form.querySelector(".filter-decision");
      submitBtn.disabled = true;
      submitBtn.textContent = "Checking...";
      progress.style.display = "block";
      status.style.display = "none";
      decision.style.display = "none";

      try {
        const response = await fetch(cfg.filterUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            entry_type: form.elements.entry_type.value,
            text: form.elements.text.value,
          }),
          cache: "no-store",
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
          throw new Error(data.error || "The entry could not be checked.");
        }

        const blocked = data.blocked === true;
        decision.className = "filter-decision mt-3 " + (blocked ? "blocked" : "allowed");
        decision.querySelector(".filter-decision-title").textContent =
          blocked ? "Entry blocked" : "Entry allowed";
        decision.querySelector(".filter-decision-reason").textContent =
          blocked ? data.reason : "This entry meets the selected guidelines.";
        decision.querySelector(".filter-json").textContent = JSON.stringify(
          { blocked: blocked, reason: blocked ? data.reason : null },
          null,
          2
        );
        decision.style.display = "block";

        status.className = "status-message " + (blocked ? "status-error" : "status-success") + " cs-status";
        status.textContent = blocked
          ? "This entry cannot be submitted. Review the reason below."
          : "This entry can be submitted.";
        status.style.display = "block";
      } catch (err) {
        status.className = "status-message status-error cs-status";
        status.textContent = err.message || String(err);
        status.style.display = "block";
      } finally {
        progress.style.display = "none";
        submitBtn.disabled = false;
        submitBtn.textContent = "Check entry";
      }
    });
  });

  // --- Puzzle Collage ---
  let lastPuzzleLayout = null;

  document.querySelectorAll("[data-puzzle-panel]").forEach((button) => {
    button.addEventListener("click", () => {
      const panel = button.getAttribute("data-puzzle-panel");
      const section = button.closest("#section-puzzle-collage");
      if (!section) return;
      section.querySelectorAll("[data-puzzle-panel]").forEach((btn) => {
        btn.classList.toggle("active", btn === button);
      });
      section.querySelectorAll("[data-puzzle-view]").forEach((view) => {
        view.style.display = view.getAttribute("data-puzzle-view") === panel ? "block" : "none";
      });
    });
  });

  function renderPuzzlePieces(form, data) {
    const result = form.querySelector(".puzzle-split-result");
    const groups = form.querySelector(".puzzle-person-groups");
    const summary = form.querySelector(".puzzle-split-summary");
    const layoutField = form.querySelector(".puzzle-layout-json");
    if (!result || !groups) return;

    const byPerson = {};
    (data.pieces || []).forEach((piece) => {
      const key = String(piece.person);
      if (!byPerson[key]) byPerson[key] = [];
      byPerson[key].push(piece);
    });

    groups.innerHTML = "";
    Object.keys(byPerson)
      .sort((a, b) => Number(a) - Number(b))
      .forEach((person) => {
        const card = document.createElement("div");
        card.className = "puzzle-person-card";
        const title = document.createElement("h3");
        title.textContent = "Person " + person + " · " + byPerson[person].length + " pieces";
        card.appendChild(title);
        const grid = document.createElement("div");
        grid.className = "puzzle-piece-grid";
        byPerson[person].forEach((piece) => {
          const item = document.createElement("div");
          item.className = "puzzle-piece-item";
          const img = document.createElement("img");
          img.alt = piece.piece_id;
          img.src = cacheBust(piece.image_url || piece.local_path || (cfg.downloadPrefix + piece.output_filename));
          const label = document.createElement("span");
          label.textContent = piece.piece_id;
          const link = document.createElement("a");
          link.href = cfg.downloadPrefix + piece.output_filename;
          link.download = piece.output_filename;
          link.className = "small";
          link.textContent = "Download";
          item.appendChild(img);
          item.appendChild(label);
          item.appendChild(link);
          grid.appendChild(item);
        });
        card.appendChild(grid);
        groups.appendChild(card);
      });

    if (summary) {
      const per = data.pieces_per_person || {};
      const perText = Object.keys(per)
        .sort((a, b) => Number(a) - Number(b))
        .map((k) => "P" + k + "=" + per[k])
        .join(", ");
      summary.textContent =
        (data.message || "") +
        " Grid " + data.rows + "×" + data.cols +
        (perText ? " · " + perText : "") +
        ". Copy the layout JSON before assembling.";
    }
    lastPuzzleLayout = data.layout || null;
    if (layoutField) {
      layoutField.value = lastPuzzleLayout ? JSON.stringify(lastPuzzleLayout) : "";
    }
    const assembleLayout = document.getElementById("puzzle-assemble-layout");
    if (assembleLayout && lastPuzzleLayout) {
      assembleLayout.value = JSON.stringify(lastPuzzleLayout, null, 2);
    }
    result.style.display = "block";
  }

  document.querySelectorAll("form[data-puzzle-split-form]").forEach((form) => {
    form.querySelectorAll('input[type="file"]').forEach((input) => {
      input.addEventListener("change", () => rememberFileInput(form, input));
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (form.dataset.generating === "1") return;
      restoreCachedFiles(form);
      if (!form.checkValidity()) {
        form.reportValidity();
        return;
      }

      const submitBtn = form.querySelector(".convert-btn");
      const progress = form.querySelector(".cs-progress");
      const status = form.querySelector(".cs-status");
      const result = form.querySelector(".puzzle-split-result");

      setBusy(form, submitBtn, true);
      if (submitBtn) submitBtn.textContent = "Splitting...";
      if (status) status.style.display = "none";
      if (result) result.style.display = "none";
      if (progress) progress.style.display = "block";

      try {
        const body = new FormData(form);
        body.set("_regen", String(Date.now()));
        const response = await fetch(cfg.puzzleSplitUrl, {
          method: "POST",
          body: body,
          cache: "no-store",
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
          throw new Error(data.error || "Split failed");
        }
        renderPuzzlePieces(form, data);
        if (status) {
          status.className = "status-message status-success cs-status";
          status.textContent = data.message || "Pieces ready.";
          status.style.display = "block";
        }
        form.dataset.hasResult = "1";
      } catch (err) {
        if (status) {
          status.className = "status-message status-error cs-status";
          status.textContent = err.message || String(err);
          status.style.display = "block";
        }
      } finally {
        if (progress) progress.style.display = "none";
        setBusy(form, submitBtn, false);
        if (submitBtn) {
          submitBtn.textContent = form.dataset.hasResult === "1" ? "Split again" : "Split into puzzle pieces";
        }
      }
    });
  });

  document.querySelectorAll("[data-puzzle-copy-layout]").forEach((button) => {
    button.addEventListener("click", async () => {
      const form = button.closest("form");
      const layoutField = form && form.querySelector(".puzzle-layout-json");
      const text = (layoutField && layoutField.value) || (lastPuzzleLayout ? JSON.stringify(lastPuzzleLayout) : "");
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied!";
        setTimeout(() => {
          button.textContent = "Copy layout JSON";
        }, 1500);
      } catch (_err) {
        if (layoutField) {
          layoutField.hidden = false;
          layoutField.select();
        }
      }
    });
  });

  document.querySelectorAll("form[data-puzzle-assemble-form]").forEach((form) => {
    form.querySelectorAll('input[type="file"]').forEach((input) => {
      input.addEventListener("change", () => rememberFileInput(form, input));
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (form.dataset.generating === "1") return;
      restoreCachedFiles(form);
      if (!form.checkValidity()) {
        form.reportValidity();
        return;
      }

      const submitBtn = form.querySelector(".convert-btn");
      const progress = form.querySelector(".cs-progress");
      const status = form.querySelector(".cs-status");
      const result = form.querySelector(".cs-result");

      setBusy(form, submitBtn, true);
      if (submitBtn) submitBtn.textContent = "Assembling...";
      if (status) status.style.display = "none";
      if (progress) progress.style.display = "block";

      try {
        const body = new FormData(form);
        body.set("_regen", String(Date.now()));
        Object.entries(fileCache(form)).forEach(([name, files]) => {
          if (!files || !files.length) return;
          const current = body.getAll(name).filter((value) => value instanceof File && value.size);
          if (current.length) return;
          body.delete(name);
          files.forEach((file) => body.append(name, file, file.name));
        });
        const response = await fetch(cfg.puzzleAssembleUrl, {
          method: "POST",
          body: body,
          cache: "no-store",
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
          throw new Error(data.error || "Assemble failed");
        }
        const img = form.querySelector(".cs-result-image");
        const download = form.querySelector(".cs-download");
        const message = form.querySelector(".cs-result-message");
        const imageUrl = data.image_url || data.local_path || (cfg.downloadPrefix + data.output_filename);
        if (img) img.src = cacheBust(imageUrl);
        if (download) {
          download.href = cfg.downloadPrefix + data.output_filename;
          download.setAttribute("download", data.output_filename);
        }
        if (message) message.textContent = data.message || "Collage assembled successfully.";
        if (result) result.style.display = "block";
        form.dataset.hasResult = "1";
        if (status) {
          status.className = "status-message status-success cs-status";
          status.textContent = data.message || "Done.";
          status.style.display = "block";
        }
      } catch (err) {
        if (status) {
          status.className = "status-message status-error cs-status";
          status.textContent = err.message || String(err);
          status.style.display = "block";
        }
      } finally {
        if (progress) progress.style.display = "none";
        setBusy(form, submitBtn, false);
        if (submitBtn) {
          submitBtn.textContent = form.dataset.hasResult === "1" ? "Assemble again" : "Assemble collage";
        }
      }
    });
  });

  const params = new URLSearchParams(window.location.search);
  const deep = params.get("station");
  if (deep && document.getElementById("section-" + deep)) {
    switchTab(deep);
  }
})();
