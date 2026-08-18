(function initRealtimeExperienceMode(global) {
  "use strict";

  const ALL_MODEL_KEYS = Object.freeze(["minwm", "lingbot2", "happyoyster"]);

  function isZingOnly(config = {}) {
    return config?.zingOnly === true;
  }

  function selectedModelKeys(config = {}, requested = []) {
    if (isZingOnly(config)) return ["minwm"];
    return Array.isArray(requested) ? requested.slice() : [];
  }

  function recordingVariants(config = {}) {
    return isZingOnly(config) ? ["zing"] : ["comparison", "zing"];
  }

  function applyToDocument(documentRef, config = {}) {
    if (!isZingOnly(config) || !documentRef) return false;
    if (documentRef.documentElement?.dataset) {
      documentRef.documentElement.dataset.realtimeExperience = "zing-only";
      documentRef.documentElement.dataset.realtimeExperienceReady = "true";
    }
    if (typeof documentRef.title === "string") {
      documentRef.title = "World Studio · Zing 实时世界";
    }
    for (const element of documentRef.querySelectorAll?.("[data-zing-only-hide]") || []) {
      element.hidden = true;
      element.setAttribute?.("aria-hidden", "true");
    }
    for (const element of documentRef.querySelectorAll?.("[data-zing-only-copy]") || []) {
      element.textContent = String(element.dataset?.zingOnlyCopy || "");
    }
    for (const element of documentRef.querySelectorAll?.("[data-zing-only-aria-label]") || []) {
      element.setAttribute?.(
        "aria-label",
        String(element.dataset?.zingOnlyAriaLabel || ""),
      );
    }
    for (const element of documentRef.querySelectorAll?.("[data-zing-only-title]") || []) {
      element.setAttribute?.("title", String(element.dataset?.zingOnlyTitle || ""));
    }
    for (const element of documentRef.querySelectorAll?.("[data-model-key]") || []) {
      if (element.dataset?.modelKey !== "minwm") {
        element.hidden = true;
        element.setAttribute?.("aria-hidden", "true");
      }
    }
    return true;
  }

  const api = Object.freeze({
    ALL_MODEL_KEYS,
    applyToDocument,
    isZingOnly,
    recordingVariants,
    selectedModelKeys,
  });
  global.SGLangRealtimeExperienceMode = api;
  if (global.document && isZingOnly(global.SGLANG_REALTIME_UI_CONFIG)) {
    if (global.document.documentElement?.dataset) {
      global.document.documentElement.dataset.realtimeExperience = "zing-only";
    }
    global.document.addEventListener?.(
      "DOMContentLoaded",
      () => applyToDocument(global.document, global.SGLANG_REALTIME_UI_CONFIG),
      { once: true },
    );
  }
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : window);
