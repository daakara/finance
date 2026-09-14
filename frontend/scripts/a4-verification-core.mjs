/**
 * a4-verification-core.mjs
 * Shared validation core for Phase A4 accessibility runtime verification.
 * Enforces strict WCAG 2.2 AA criteria across both positive audit and negative control suites.
 */

export function calculateLuminance(r, g, b) {
  const [rs, gs, bs] = [r, g, b].map((c) => {
    c = c / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
}

export function parseRgb(rgbStr) {
  if (!rgbStr) return [0, 0, 0];
  const match = rgbStr.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
  if (match) {
    return [parseInt(match[1], 10), parseInt(match[2], 10), parseInt(match[3], 10)];
  }
  if (rgbStr.startsWith("#")) {
    let hex = rgbStr.replace("#", "");
    if (hex.length === 3) hex = hex.split("").map((c) => c + c).join("");
    const num = parseInt(hex, 16);
    return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
  }
  return [0, 0, 0];
}

export function calculateContrastRatio(fgRgb, bgRgb) {
  const l1 = calculateLuminance(...fgRgb);
  const l2 = calculateLuminance(...bgRgb);
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}

/**
 * Determine WCAG 2.2 AA required contrast ratio.
 * Large text: >= 24px (18pt) OR (>= 18.66px (14pt) AND bold >= 700) -> 3.0:1
 * Normal text: all other text -> 4.5:1
 */
export function getRequiredContrastRatio(fontSizePx, fontWeight) {
  const isLarge = fontSizePx >= 24 || (fontSizePx >= 18.66 && fontWeight >= 700);
  return isLarge ? 3.0 : 4.5;
}

/**
 * Check reflow metrics (WCAG 1.4.10) for horizontal scrolling and bounding bounds.
 */
export async function checkReflowMetrics(page) {
  return await page.evaluate(() => {
    const docEl = document.documentElement;
    const body = document.body;
    const scrollWidth = Math.max(docEl.scrollWidth, body.scrollWidth);
    const clientWidth = docEl.clientWidth;
    const innerWidth = window.innerWidth;
    const hasHorizontalScroll = scrollWidth > clientWidth + 1;

    // Check for clipped text elements
    const allTextEls = Array.from(document.querySelectorAll("p, span, h1, h2, h3, h4, h5, h6, th, td"));
    let clippedTextCount = 0;
    const clippedSamples = [];

    for (const el of allTextEls) {
      if (!el.offsetParent && el.tagName !== "BODY") continue;
      const cs = window.getComputedStyle(el);
      const isEllipsis = cs.textOverflow === "ellipsis";
      if (
        el.scrollWidth > el.clientWidth + 2 &&
        cs.overflow === "hidden" &&
        !isEllipsis &&
        !el.closest("table") &&
        !el.closest('[role="tablist"]') &&
        !el.closest(".overflow-x-auto")
      ) {
        clippedTextCount++;
        if (clippedSamples.length < 5) {
          clippedSamples.push({
            tag: el.tagName,
            id: el.id,
            text: (el.textContent || "").trim().substring(0, 40),
            scrollWidth: el.scrollWidth,
            clientWidth: el.clientWidth,
          });
        }
      }
    }

    // Check focusable controls out of horizontal bounds
    const focusableControls = Array.from(
      document.querySelectorAll("button, input, select, textarea, a[href]")
    );
    let outOfBoundsControlsCount = 0;
    let visibleControlsCount = 0;

    for (const el of focusableControls) {
      if (
        el.closest("table") ||
        el.closest('[role="tablist"]') ||
        el.closest(".overflow-x-auto")
      ) {
        continue;
      }
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        visibleControlsCount++;
        if (rect.right > clientWidth + 5 || rect.left < -5) {
          outOfBoundsControlsCount++;
        }
      }
    }

    return {
      scrollWidth,
      clientWidth,
      innerWidth,
      hasHorizontalScroll,
      clippedTextCount,
      clippedSamples,
      totalControls: focusableControls.length,
      visibleControlsCount,
      outOfBoundsControlsCount,
    };
  });
}

/**
 * Check exact focus restoration after dialog dismissal.
 * Enforces that activeElement matches the exact trigger element (or its descendant),
 * explicitly rejecting generic button/input focus on unrelated elements.
 */
export async function checkExactFocusRestoration(page, triggerMarkerAttr) {
  return await page.evaluate((marker) => {
    const active = document.activeElement;
    const triggerEl = document.querySelector(`[${marker}]`);
    if (!triggerEl) {
      return {
        matched: false,
        reason: `Trigger element with attribute [${marker}] not found in document`,
        activeTag: active ? active.tagName : null,
        activeId: active ? active.id : null,
      };
    }

    const isExact = active === triggerEl || triggerEl.contains(active);
    return {
      matched: isExact,
      reason: isExact ? "Exact trigger focus restored" : "Focus not restored to exact trigger",
      triggerTag: triggerEl.tagName,
      triggerId: triggerEl.id || null,
      triggerAriaLabel: triggerEl.getAttribute("aria-label") || null,
      activeTag: active ? active.tagName : null,
      activeId: active ? active.id : null,
      activeAriaLabel: active ? active.getAttribute("aria-label") : null,
      activeText: active ? (active.textContent || "").trim().substring(0, 30) : null,
    };
  }, triggerMarkerAttr);
}

/**
 * Comprehensive runtime contrast audit across all visible text-bearing elements.
 * Strictly enforces 4.5:1 for normal text and 3.0:1 for large text.
 * Performs true alpha-compositing across layered background fills.
 */
export async function auditVisibleElementsContrast(page, themeName) {
  const auditData = await page.evaluate((tName) => {
    function parseRgba(str) {
      if (!str) return null;
      const m = str.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
      if (m) {
        return [
          parseInt(m[1], 10),
          parseInt(m[2], 10),
          parseInt(m[3], 10),
          m[4] !== undefined ? parseFloat(m[4]) : 1,
        ];
      }
      return null;
    }

    function getEffectiveBg(element) {
      const baseRgb = tName.includes("Paper") ? [248, 250, 252] : [9, 13, 20];
      const path = [];
      let curr = element;
      while (curr && curr.nodeType === Node.ELEMENT_NODE) {
        path.unshift(curr);
        curr = curr.parentElement;
      }
      let currentRgb = [...baseRgb];
      for (const node of path) {
        const cs = window.getComputedStyle(node);
        const parsed = parseRgba(cs.backgroundColor);
        if (parsed && parsed[3] > 0) {
          const a = parsed[3];
          currentRgb = [
            Math.round(parsed[0] * a + currentRgb[0] * (1 - a)),
            Math.round(parsed[1] * a + currentRgb[1] * (1 - a)),
            Math.round(parsed[2] * a + currentRgb[2] * (1 - a)),
          ];
        }
      }
      return `rgb(${currentRgb[0]}, ${currentRgb[1]}, ${currentRgb[2]})`;
    }

    const elements = Array.from(
      document.querySelectorAll("h1, h2, h3, h4, h5, h6, p, th, td, a, button, label, [role='tab']")
    );

    const audited = [];
    for (const el of elements) {
      // Must be visible
      if (!el.offsetParent && el.tagName !== "BODY") continue;
      const cs = window.getComputedStyle(el);
      if (cs.visibility === "hidden" || cs.display === "none") continue;
      const text = (el.textContent || "").trim();
      if (!text || text.length === 0) continue;

      const fg = cs.color;
      const bg = getEffectiveBg(el);
      const fontSize = parseFloat(cs.fontSize) || 14;
      const fontWeight = parseInt(cs.fontWeight, 10) || 400;
      const isLarge = fontSize >= 24 || (fontSize >= 18.66 && fontWeight >= 700);
      const requiredRatio = isLarge ? 3.0 : 4.5;

      audited.push({
        tag: el.tagName.toLowerCase(),
        id: el.id || null,
        className: (el.className || "").toString().substring(0, 50),
        textSnippet: text.substring(0, 40),
        fg,
        bg,
        fontSize,
        fontWeight,
        isLarge,
        requiredRatio,
      });
    }

    return audited;
  }, themeName);

  let passedCount = 0;
  let failedCount = 0;
  const violations = [];
  const sampleResults = [];

  for (const item of auditData) {
    const fgRgb = parseRgb(item.fg);
    const bgRgb = parseRgb(item.bg);
    const ratio = calculateContrastRatio(fgRgb, bgRgb);
    const pass = ratio >= item.requiredRatio;

    if (pass) {
      passedCount++;
    } else {
      failedCount++;
      violations.push({
        ...item,
        actualRatio: Number(ratio.toFixed(2)),
      });
    }

    if (sampleResults.length < 25) {
      sampleResults.push({
        ...item,
        actualRatio: Number(ratio.toFixed(2)),
        pass,
      });
    }
  }

  return {
    totalAudited: auditData.length,
    passedCount,
    failedCount,
    violations,
    sampleResults,
  };
}

/**
 * Non-Text Contrast Audit (WCAG 2.2 Success Criterion 1.4.11, Level AA).
 * Enforces >= 3.0:1 contrast ratio for:
 * 1. Visual keyboard focus indicators (outline / boxShadow) on interactive controls
 * 2. Active form input boundaries (inputs, selects, textareas)
 * 3. Active / selected UI state indicators (selected tabs)
 */
export async function auditNonTextContrast(page, themeName) {
  const auditData = await page.evaluate((tName) => {
    function parseRgba(str) {
      if (!str) return null;
      const m = str.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
      if (m) {
        return [
          parseInt(m[1], 10),
          parseInt(m[2], 10),
          parseInt(m[3], 10),
          m[4] !== undefined ? parseFloat(m[4]) : 1,
        ];
      }
      return null;
    }

    function calculateLuminance(r, g, b) {
      const [rs, gs, bs] = [r, g, b].map((c) => {
        c = c / 255;
        return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
      });
      return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
    }

    function getContrast(fg, bg) {
      const l1 = calculateLuminance(fg[0], fg[1], fg[2]);
      const l2 = calculateLuminance(bg[0], bg[1], bg[2]);
      const lighter = Math.max(l1, l2);
      const darker = Math.min(l1, l2);
      return (lighter + 0.05) / (darker + 0.05);
    }

    function getEffectiveBg(element) {
      const baseRgb = tName.toLowerCase().includes("paper") ? [248, 250, 252] : [9, 13, 20];
      const path = [];
      let curr = element;
      while (curr && curr.nodeType === Node.ELEMENT_NODE) {
        path.unshift(curr);
        curr = curr.parentElement;
      }
      let currentRgb = [...baseRgb];
      for (const node of path) {
        const cs = window.getComputedStyle(node);
        const parsed = parseRgba(cs.backgroundColor);
        if (parsed && parsed[3] > 0) {
          const a = parsed[3];
          currentRgb = [
            Math.round(parsed[0] * a + currentRgb[0] * (1 - a)),
            Math.round(parsed[1] * a + currentRgb[1] * (1 - a)),
            Math.round(parsed[2] * a + currentRgb[2] * (1 - a)),
          ];
        }
      }
      return `rgb(${currentRgb[0]}, ${currentRgb[1]}, ${currentRgb[2]})`;
    }

    const controls = Array.from(
      document.querySelectorAll("input:not([type='hidden']), select, textarea, button, a[href], [role='tab']")
    );

    const audited = [];

    for (const el of controls) {
      if (!el.offsetParent && el.tagName !== "BODY") continue;
      if (el.disabled || el.getAttribute("aria-disabled") === "true") continue;
      const initialCs = window.getComputedStyle(el);
      if (initialCs.visibility === "hidden" || initialCs.display === "none") continue;

      const effectiveBg = getEffectiveBg(el.parentElement || el);
      const bgRgb = parseRgba(effectiveBg) || [9, 13, 20];
      const isInput = ["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName);

      // Focus element to evaluate active focus indicators and active state
      el.focus();
      const focusCs = window.getComputedStyle(el);

      // 1. Visible Keyboard Focus Indicator
      // An interactive component can indicate focus via outline, box-shadow (focus ring), or border change.
      // Evaluate all active indicators and select the best one (highest contrast against background).
      let bestFocusRatio = 0;
      let focusIndicatorColor = null;
      let focusIndicatorSource = null;

      const outlineWidth = parseFloat(focusCs.outlineWidth) || 0;
      const outlineStyle = focusCs.outlineStyle;
      const parsedOutline = parseRgba(focusCs.outlineColor);

      if (outlineWidth > 0 && outlineStyle !== "none" && parsedOutline && parsedOutline[3] > 0.1) {
        const r = getContrast(parsedOutline, bgRgb);
        if (r > bestFocusRatio) {
          bestFocusRatio = r;
          focusIndicatorColor = focusCs.outlineColor;
          focusIndicatorSource = "outline";
        }
      }

      if (focusCs.boxShadow && focusCs.boxShadow !== "none") {
        const shadowMatches = [...focusCs.boxShadow.matchAll(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/g)];
        for (const m of shadowMatches) {
          const p = parseRgba(m[0]);
          if (p && p[3] > 0.1) {
            const ratio = getContrast(p, bgRgb);
            if (ratio > bestFocusRatio) {
              bestFocusRatio = ratio;
              focusIndicatorColor = m[0];
              focusIndicatorSource = "boxShadow";
            }
          }
        }
      }

      if (focusCs.borderColor !== initialCs.borderColor) {
        const parsedBorder = parseRgba(focusCs.borderColor);
        if (parsedBorder && parsedBorder[3] > 0.1) {
          const r = getContrast(parsedBorder, bgRgb);
          if (r > bestFocusRatio) {
            bestFocusRatio = r;
            focusIndicatorColor = focusCs.borderColor;
            focusIndicatorSource = "borderColor";
          }
        }
      }

      // If no indicator had > 0 contrast, still capture outline or border if present so failures are reported
      if (!focusIndicatorColor && outlineWidth > 0 && outlineStyle !== "none" && parsedOutline && parsedOutline[3] > 0.1) {
        focusIndicatorColor = focusCs.outlineColor;
        focusIndicatorSource = "outline";
      }

      if (focusIndicatorColor) {
        audited.push({
          tag: el.tagName.toLowerCase(),
          id: el.id || null,
          className: (el.className || "").toString().substring(0, 50),
          componentType: isInput ? "form_input_focus_indicator" : "ui_control_focus_indicator",
          indicatorColor: focusIndicatorColor,
          indicatorSource: focusIndicatorSource,
          effectiveBg,
          requiredRatio: 3.0,
        });
      }

      // 2. Active Form Input Boundary
      if (isInput) {
        let boundaryColor = focusCs.borderColor;
        let boundarySource = "borderColor";
        let boundaryRatio = 0;
        const parsedBorder = parseRgba(boundaryColor);
        if (parsedBorder && parsedBorder[3] > 0.1) {
          boundaryRatio = getContrast(parsedBorder, bgRgb);
        }

        // If the focus outline or focus ring provides a higher-contrast boundary, consider that boundary
        if (bestFocusRatio > boundaryRatio && focusIndicatorColor) {
          boundaryColor = focusIndicatorColor;
          boundarySource = focusIndicatorSource;
          boundaryRatio = bestFocusRatio;
        }

        if (boundaryColor) {
          audited.push({
            tag: el.tagName.toLowerCase(),
            id: el.id || null,
            className: (el.className || "").toString().substring(0, 50),
            componentType: "active_form_input_boundary",
            indicatorColor: boundaryColor,
            indicatorSource: boundarySource,
            effectiveBg,
            requiredRatio: 3.0,
          });
        }
      }

      // 3. Active / Selected Tab Indicator
      if (el.getAttribute("role") === "tab" && el.getAttribute("aria-selected") === "true") {
        const tabBorder = focusCs.borderBottomColor || focusCs.borderColor;
        const tabBorderWidth = parseFloat(focusCs.borderBottomWidth || focusCs.borderWidth) || 0;
        const parsedTabBorder = parseRgba(tabBorder);
        if (tabBorderWidth > 0 && parsedTabBorder && parsedTabBorder[3] > 0.1) {
          audited.push({
            tag: "tab[selected]",
            id: el.id || null,
            className: (el.className || "").toString().substring(0, 50),
            componentType: "active_tab_selected_indicator",
            indicatorColor: tabBorder,
            indicatorSource: "borderBottomColor",
            effectiveBg,
            requiredRatio: 3.0,
          });
        }
      }

      el.blur();
    }

    return audited;
  }, themeName);

  let passedCount = 0;
  let failedCount = 0;
  const violations = [];
  const sampleResults = [];

  for (const item of auditData) {
    const fgRgb = parseRgb(item.indicatorColor);
    const bgRgb = parseRgb(item.effectiveBg);
    const ratio = calculateContrastRatio(fgRgb, bgRgb);
    const pass = ratio >= item.requiredRatio;

    if (pass) {
      passedCount++;
    } else {
      failedCount++;
      violations.push({
        ...item,
        actualRatio: Number(ratio.toFixed(2)),
      });
    }

    if (sampleResults.length < 25) {
      sampleResults.push({
        ...item,
        actualRatio: Number(ratio.toFixed(2)),
        pass,
      });
    }
  }

  return {
    totalAudited: auditData.length,
    passedCount,
    failedCount,
    violations,
    sampleResults,
  };
}

