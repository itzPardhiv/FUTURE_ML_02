const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const url = 'http://localhost:8501/';
  await page.goto(url, { waitUntil: 'networkidle' });

  const pages = [
    { label: '📊 Dashboard', file: 'dashboard_overview.png' },
    { label: '🎫 Triage', file: 'ticket_triage.png' },
    { label: '📈 Performance', file: 'model_performance.png' },
    { label: '🗂 Dataset', file: 'dataset_explorer.png' },
    { label: 'ℹ About', file: 'about_project.png' },
  ];

  for (const item of pages) {
    try {
      // Click the sidebar radio by accessible name (emoji + label)
      await page.getByRole('radio', { name: item.label }).first().click({ timeout: 5000 });
    } catch (e) {
      // Fallback: click any element containing the label text
      try {
        await page.locator(`text=${item.label}`).first().click({ timeout: 3000 });
      } catch (e2) {
        console.log('Could not click label for', item.label, e2.message);
      }
    }
    // Wait for the view to update before screenshot
    await page.waitForTimeout(1200);
    // Ensure the visuals directory exists relative to project root
    const outDir = path.join(__dirname, '..', 'visuals');
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
    await page.screenshot({ path: path.join(outDir, item.file), fullPage: true });
    console.log('Saved', item.file);
  }

  await browser.close();
})();
