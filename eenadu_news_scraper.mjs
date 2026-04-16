import { chromium } from '/opt/node22/lib/node_modules/playwright/index.mjs';

const STEEL_SESSION_ID = 'ed6f566a-1af3-4ce8-9d3e-c0dad8901b13';
const CDP_ENDPOINT = `wss://connect.steel.dev?sessionId=${STEEL_SESSION_ID}`;
const TARGET_URL = 'https://www.eenadu.net/movies';

async function scrapeEenaduMoviesNews() {
  console.log(`Connecting to Steel.dev session: ${STEEL_SESSION_ID}`);
  console.log(`CDP Endpoint: ${CDP_ENDPOINT}\n`);

  let browser;
  try {
    browser = await chromium.connectOverCDP(CDP_ENDPOINT, { timeout: 30000 });
    console.log('Connected to Steel browser successfully.');

    const contexts = browser.contexts();
    const context = contexts.length > 0 ? contexts[0] : await browser.newContext();
    const pages = context.pages();
    const page = pages.length > 0 ? pages[0] : await context.newPage();

    console.log(`Navigating to ${TARGET_URL} ...\n`);
    await page.goto(TARGET_URL, { waitUntil: 'domcontentloaded', timeout: 30000 });

    // Wait for news content to load
    await page.waitForTimeout(3000);

    const pageTitle = await page.title();
    console.log(`Page Title: ${pageTitle}\n`);

    // Scrape news headlines and links from eenadu.net/movies
    const news = await page.evaluate(() => {
      const results = [];

      // Try various selectors commonly used by Eenadu for news items
      const selectors = [
        'article',
        '.news-item',
        '.story-item',
        '.top-story',
        '.district-top-story',
        'h2 a',
        'h3 a',
        '.headline a',
        '.news-head a',
        '.news-title a',
        '.eng-news a',
        '[class*="story"] a',
        '[class*="news"] a',
        '[class*="headline"] a',
      ];

      // Collect all anchor tags that look like news links
      const seen = new Set();
      for (const sel of selectors) {
        const elements = document.querySelectorAll(sel);
        for (const el of elements) {
          const anchor = el.tagName === 'A' ? el : el.querySelector('a');
          if (!anchor) continue;
          const href = anchor.href || '';
          const text = anchor.innerText?.trim() || el.innerText?.trim() || '';
          if (
            text.length > 10 &&
            href &&
            !seen.has(href) &&
            (href.includes('eenadu') || href.startsWith('/'))
          ) {
            seen.add(href);
            results.push({ headline: text.replace(/\s+/g, ' ').substring(0, 200), url: href });
          }
        }
      }

      // If nothing found, fall back to all meaningful anchor tags
      if (results.length === 0) {
        document.querySelectorAll('a').forEach((a) => {
          const text = a.innerText?.trim() || '';
          const href = a.href || '';
          if (
            text.length > 15 &&
            href &&
            !seen.has(href) &&
            (href.includes('/movies') || href.includes('/telugu'))
          ) {
            seen.add(href);
            results.push({ headline: text.replace(/\s+/g, ' ').substring(0, 200), url: href });
          }
        });
      }

      return results.slice(0, 30); // Return top 30 news items
    });

    console.log(`=== Eenadu Movies - Latest News (${news.length} items) ===\n`);
    if (news.length === 0) {
      console.log('No news items found with standard selectors. Dumping visible text...\n');
      const bodyText = await page.evaluate(() => {
        return document.body.innerText.substring(0, 3000);
      });
      console.log(bodyText);
    } else {
      news.forEach((item, idx) => {
        console.log(`${idx + 1}. ${item.headline}`);
        console.log(`   URL: ${item.url}\n`);
      });
    }

    return news;
  } finally {
    if (browser) {
      await browser.close();
      console.log('\nBrowser connection closed.');
    }
  }
}

scrapeEenaduMoviesNews().catch((err) => {
  console.error('Error:', err.message);
  process.exit(1);
});
