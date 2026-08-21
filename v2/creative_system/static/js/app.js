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
    if (progress) progress.style.display = "block";

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

  const params = new URLSearchParams(window.location.search);
  const deep = params.get("station");
  if (deep && document.getElementById("section-" + deep)) {
    switchTab(deep);
  }
})();
