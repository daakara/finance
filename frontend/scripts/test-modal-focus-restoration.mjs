import puppeteer from "puppeteer";

async function run() {
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu"],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  await page.goto("http://localhost:3000/", { waitUntil: "load" });
  await new Promise(r => setTimeout(r, 2500));

  console.log("=== 1. PRIVACY MODAL FOCUS RESTORATION ===");
  const privacyTriggerInfo = await page.evaluate(() => {
    const el = document.getElementById("privacy-settings-btn");
    if (!el) return null;
    return {
      visibleLabel: el.innerText.trim() || el.getAttribute("aria-label") || el.title,
      selector: "#privacy-settings-btn",
      routeLocation: "Global header[role=\"banner\"] toolbar",
      eventDispatched: "click -> setIsPrivacyOpen(true)",
      tagName: el.tagName,
      id: el.id,
      ariaLabel: el.getAttribute("aria-label")
    };
  });
  console.log("Privacy Trigger:", JSON.stringify(privacyTriggerInfo, null, 2));

  // Focus trigger
  await page.evaluate(() => document.getElementById("privacy-settings-btn").focus());
  const privacyFocusBefore = await page.evaluate(() => `${document.activeElement.tagName}#${document.activeElement.id}`);
  console.log("Focus before open:", privacyFocusBefore);

  // Click trigger
  await page.click("#privacy-settings-btn");
  await page.waitForSelector('[role="dialog"][aria-label="Privacy and Data Telemetry Settings"]', { visible: true });
  await new Promise(r => setTimeout(r, 60));

  const privacyInitialFocus = await page.evaluate(() => `${document.activeElement.tagName}#${document.activeElement.id}`);
  console.log("Initial dialog focus:", privacyInitialFocus);

  // Tab forward through focusables
  await page.keyboard.press("Tab");
  const privacyTab1 = await page.evaluate(() => `${document.activeElement.tagName}#${document.activeElement.id}`);
  console.log("Focus after Tab 1:", privacyTab1);

  await page.keyboard.press("Tab");
  const privacyTab2 = await page.evaluate(() => `${document.activeElement.tagName}#${document.activeElement.id}`);
  console.log("Focus after Tab 2:", privacyTab2);

  // Shift+Tab backward
  await page.keyboard.down("Shift");
  await page.keyboard.press("Tab");
  await page.keyboard.up("Shift");
  const privacyShiftTab = await page.evaluate(() => `${document.activeElement.tagName}#${document.activeElement.id}`);
  console.log("Focus after Shift+Tab:", privacyShiftTab);

  // Press Escape
  await page.keyboard.press("Escape");
  await page.waitForSelector('[role="dialog"][aria-label="Privacy and Data Telemetry Settings"]', { hidden: true });
  await new Promise(r => setTimeout(r, 60));

  const privacyFocusAfterEscape = await page.evaluate(() => `${document.activeElement.tagName}#${document.activeElement.id}`);
  console.log("Focus after Escape:", privacyFocusAfterEscape);
  const privacyPass = (privacyFocusAfterEscape === "BUTTON#privacy-settings-btn");
  console.log("Privacy restoration valid:", privacyPass ? "PASS" : "FAIL");


  console.log("\n=== 2. ONBOARDING TOUR MODAL FOCUS RESTORATION ===");
  const obTriggerInfo = await page.evaluate(() => {
    const el = document.getElementById("onboarding-tour-btn");
    if (!el) return null;
    return {
      visibleLabel: el.innerText.trim() || el.getAttribute("aria-label") || el.title,
      selector: "#onboarding-tour-btn",
      routeLocation: "Global header[role=\"banner\"] toolbar",
      eventDispatched: "click -> handleOpenOnboarding -> setIsOnboardingOpen(true)",
      tagName: el.tagName,
      id: el.id,
      ariaLabel: el.getAttribute("aria-label")
    };
  });
  console.log("Onboarding Trigger:", JSON.stringify(obTriggerInfo, null, 2));

  // Focus trigger
  await page.evaluate(() => document.getElementById("onboarding-tour-btn").focus());
  const obFocusBefore = await page.evaluate(() => `${document.activeElement.tagName}#${document.activeElement.id}`);
  console.log("Focus before open:", obFocusBefore);

  // Click trigger
  await page.click("#onboarding-tour-btn");
  await page.waitForSelector('[role="dialog"][aria-labelledby="tour-modal-title"]', { visible: true });
  await new Promise(r => setTimeout(r, 60));

  const obInitialFocus = await page.evaluate(() => `${document.activeElement.tagName}#${document.activeElement.id}`);
  console.log("Initial dialog focus:", obInitialFocus);

  // Tab forward through focusables
  await page.keyboard.press("Tab");
  const obTab1 = await page.evaluate(() => `${document.activeElement.tagName}#${document.activeElement.id}`);
  console.log("Focus after Tab 1:", obTab1);

  // Shift+Tab backward
  await page.keyboard.down("Shift");
  await page.keyboard.press("Tab");
  await page.keyboard.up("Shift");
  const obShiftTab = await page.evaluate(() => `${document.activeElement.tagName}#${document.activeElement.id}`);
  console.log("Focus after Shift+Tab:", obShiftTab);

  // Press Escape
  await page.keyboard.press("Escape");
  await page.waitForSelector('[role="dialog"][aria-labelledby="tour-modal-title"]', { hidden: true });
  await new Promise(r => setTimeout(r, 60));

  const obFocusAfterEscape = await page.evaluate(() => `${document.activeElement.tagName}#${document.activeElement.id}`);
  console.log("Focus after Escape:", obFocusAfterEscape);
  const obPass = (obFocusAfterEscape === "BUTTON#onboarding-tour-btn");
  console.log("Onboarding restoration valid:", obPass ? "PASS" : "FAIL");

  await browser.close();

  const finalOutput = {
    privacy: {
      trigger: privacyTriggerInfo,
      focusBeforeOpen: privacyFocusBefore,
      initialDialogFocus: privacyInitialFocus,
      focusAfterTab1: privacyTab1,
      focusAfterTab2: privacyTab2,
      focusAfterShiftTab: privacyShiftTab,
      focusAfterEscape: privacyFocusAfterEscape,
      pass: privacyPass
    },
    onboarding: {
      trigger: obTriggerInfo,
      focusBeforeOpen: obFocusBefore,
      initialDialogFocus: obInitialFocus,
      focusAfterTab1: obTab1,
      focusAfterShiftTab: obShiftTab,
      focusAfterEscape: obFocusAfterEscape,
      pass: obPass
    }
  };

  console.log("\n=== FINAL_AUDIT_DATA ===");
  console.log(JSON.stringify(finalOutput, null, 2));

  if (!privacyPass || !obPass) {
    process.exit(1);
  }
}

run().catch((e) => {
  console.error("FATAL ERROR:", e);
  process.exit(1);
});
