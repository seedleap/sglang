(function (global) {
  function createFullscreenController({
    documentRef = document,
    target,
    button,
    onError = () => {},
  }) {
    const sync = () => {
      const active = documentRef.fullscreenElement === target;
      const label = active
        ? "Exit fullscreen comparison"
        : "Enter fullscreen comparison";
      button.setAttribute("aria-pressed", String(active));
      button.setAttribute("aria-label", label);
      button.title = label;
    };

    const toggle = async () => {
      try {
        if (documentRef.fullscreenElement === target) {
          await documentRef.exitFullscreen();
        } else {
          await target.requestFullscreen();
        }
      } catch (error) {
        onError(error);
      }
    };

    button.addEventListener("click", toggle);
    documentRef.addEventListener("fullscreenchange", sync);
    sync();

    return {
      toggle,
      sync,
      destroy() {
        button.removeEventListener("click", toggle);
        documentRef.removeEventListener("fullscreenchange", sync);
      },
    };
  }

  global.SGLangFullscreen = { createFullscreenController };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { createFullscreenController };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
