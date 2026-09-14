/**
 * Computes the accessible name for an interactive element following W3C AccName 1.1 core principles.
 */
export function computeAccessibleName(el) {
  if (!el) return null;

  // 1. Check aria-labelledby
  const labelledBy = el.getAttribute("aria-labelledby");
  if (labelledBy) {
    const ids = labelledBy.trim().split(/\s+/);
    const parts = [];
    for (const id of ids) {
      const target = document.getElementById(id);
      if (target) {
        const text = (target.textContent || "").trim();
        if (text) parts.push(text);
      }
    }
    if (parts.length > 0) {
      return parts.join(" ");
    }
    // If aria-labelledby was specified but invalid or pointing to empty elements,
    // according to AccName it takes precedence over aria-label, but if completely empty, continue to next fallback
  }

  // 2. Check aria-label
  const ariaLabel = el.getAttribute("aria-label");
  if (ariaLabel && ariaLabel.trim().length > 0) {
    return ariaLabel.trim();
  }

  // 3. Check native form labeling (<label for="id"> or wrapping <label>)
  if (["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName)) {
    if (el.id) {
      const label = document.querySelector(`label[for="${el.id}"]`);
      if (label && (label.textContent || "").trim().length > 0) {
        return (label.textContent || "").trim();
      }
    }
    const parentLabel = el.closest("label");
    if (parentLabel) {
      // Clone label, remove input element to get pure text
      const clone = parentLabel.cloneNode(true);
      const inputInClone = clone.querySelector(`input, select, textarea`);
      if (inputInClone) inputInClone.remove();
      const text = (clone.textContent || "").trim();
      if (text.length > 0) return text;
    }

    // Input values for action buttons
    const type = (el.getAttribute("type") || "").toLowerCase();
    if (["button", "submit", "reset"].includes(type)) {
      const val = el.getAttribute("value");
      if (val && val.trim().length > 0) return val.trim();
    }
    if (type === "image") {
      const alt = el.getAttribute("alt");
      if (alt && alt.trim().length > 0) return alt.trim();
    }

    // Explicit check: placeholder is NOT an accessible name
    return null;
  }

  // 4. Content subtree for buttons, links, tabs, role="button"
  const TEXT_NODE = typeof Node !== "undefined" ? Node.TEXT_NODE : 3;
  const ELEMENT_NODE = typeof Node !== "undefined" ? Node.ELEMENT_NODE : 1;

  function getSubtreeText(node) {
    if (!node) return "";
    if (node.nodeType === TEXT_NODE) {
      return node.textContent || "";
    }
    if (node.nodeType === ELEMENT_NODE || !node.nodeType) {
      if (typeof node.getAttribute === "function" && node.getAttribute("aria-hidden") === "true") {
        return "";
      }
      if (node.tagName === "IMG") {
        const alt = node.getAttribute("alt");
        // alt="" means decorative
        return alt && alt.trim().length > 0 ? alt.trim() : "";
      }
      if (node.tagName === "SVG") {
        const svgAria = node.getAttribute("aria-label");
        if (svgAria && svgAria.trim().length > 0) return svgAria.trim();
        const titleEl = node.querySelector("title");
        if (titleEl && (titleEl.textContent || "").trim().length > 0) {
          return (titleEl.textContent || "").trim();
        }
        return "";
      }
      let acc = "";
      if (node.childNodes && typeof node.childNodes[Symbol.iterator] === "function") {
        for (const child of node.childNodes) {
          acc += " " + getSubtreeText(child);
        }
      } else if (node.textContent || node.innerText) {
        acc = node.textContent || node.innerText || "";
      }
      return acc;
    }
    return "";
  }

  const subtreeText = getSubtreeText(el).replace(/\s+/g, " ").trim();
  if (subtreeText.length > 0) {
    return subtreeText;
  }

  return null;
}
