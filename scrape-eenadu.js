const { chromium } = require('playwright');

const SESSION_ID = '743e65cd-d489-415f-8060-0b1a0997259e';
const WS_URL = `wss://connect.steel.dev?sessionId=${SESSION_ID}`;

(async () => {
  console.log('Connecting to Steel browser...');
  const browser = await chromium.connectOverCDP(WS_URL);

  const context = browser.contexts()[0] || await browser.newContext();
  const page = context.pages()[0] || await context.newPage();

  console.log('Navigating to eenadu.net...');
  await page.goto('https://www.eenadu.net', { waitUntil: 'domcontentloaded', timeout: 30000 });
  console.log('Loaded:', page.url());

  // Look for a Movies link
  const moviesLink = await page.locator('a').filter({ hasText: /సినిమా|movies|cinema/i }).first();
  const href = await moviesLink.getAttribute('href').catch(() => null);
  console.log('Movies link href:', href);

  if (href) {
    const moviesUrl = href.startsWith('http') ? href : `https://www.eenadu.net${href}`;
    console.log('Navigating to movies:', moviesUrl);
    await page.goto(moviesUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
  } else {
    await moviesLink.click();
    await page.waitForLoadState('domcontentloaded');
  }

  console.log('Movies page URL:', page.url());

  // Extract movie news headlines
  await page.waitForTimeout(2000);

  const headlines = await page.evaluate(() => {
    const selectors = ['h1', 'h2', 'h3', 'h4', '.story-title', '.article-title', '.news-title', 'a[href*="movie"], a[href*="cinema"]'];
    const seen = new Set();
    const results = [];

    document.querySelectorAll(selectors.join(', ')).forEach(el => {
      const text = el.textContent?.trim();
      if (text && text.length > 10 && !seen.has(text)) {
        seen.add(text);
        results.push(text);
      }
    });

    return results.slice(0, 40);
  });

  console.log('\n=== MOVIE NEWS HEADLINES ===\n');
  headlines.forEach((h, i) => console.log(`${i + 1}. ${h}`));

  await browser.close();
})();
