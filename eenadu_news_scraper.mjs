import puppeteer from './node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js';

const STEEL_API_KEY = 'ste-xNFvD9RTtjBGnsgDBcUvh9KQDr6hi1mLUhS983MbhELbNXrJNf4a6ybrq25SNE6Ls40w1vlu2dP0U1ox79XG7DWnHHJJu2a7S0Q';
const SESSION_ID   = '6ee28b5e-cb73-485e-a0b5-1569d5d9160e';
const CDP_ENDPOINT = `wss://connect.steel.dev?sessionId=${SESSION_ID}&apiKey=${STEEL_API_KEY}`;
const TARGET_URL   = 'https://www.eenadu.net/movies';

async function scrape() {
  console.log(`Connecting to Steel session ${SESSION_ID} via CDP...`);
  const browser = await puppeteer.connect({ browserWSEndpoint: CDP_ENDPOINT });
  console.log('Connected.\n');

  try {
    const pages = await browser.pages();
    const page  = pages.length > 0 ? pages[0] : await browser.newPage();

    console.log(`Navigating to ${TARGET_URL} ...`);
    await page.goto(TARGET_URL, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await new Promise(r => setTimeout(r, 3000));

    const title = await page.title();
    console.log(`Page title: ${title}\n`);

    const news = await page.evaluate(() => {
      const seen    = new Set();
      const results = [];

      const push = (text, href) => {
        text = text.trim().replace(/\s+/g, ' ');
        if (text.length > 10 && href && !seen.has(href)) {
          seen.add(href);
          results.push({ headline: text.substring(0, 200), url: href });
        }
      };

      // Primary selectors for Eenadu news items
      const selectors = [
        'article a', 'h2 a', 'h3 a', 'h4 a',
        '.news-item a', '.story-item a', '.top-story a',
        '.headline a', '.news-head a', '.news-title a',
        '[class*="story"] a', '[class*="news"] a', '[class*="headline"] a',
        '.eng-news a', '.carousel a', '.slider a',
      ];

      for (const sel of selectors) {
        document.querySelectorAll(sel).forEach(a => {
          push(a.innerText || a.textContent || '', a.href || '');
        });
      }

      // Fallback: any anchor with text length > 15
      if (results.length === 0) {
        document.querySelectorAll('a').forEach(a => {
          push(a.innerText || a.textContent || '', a.href || '');
        });
      }

      return results.slice(0, 30);
    });

    console.log(`=== Eenadu Movies — Latest News (${news.length} items) ===\n`);
    if (news.length === 0) {
      const body = await page.evaluate(() => document.body.innerText.substring(0, 3000));
      console.log('(No structured items found)\n', body);
    } else {
      news.forEach((item, i) => {
        console.log(`${i + 1}. ${item.headline}`);
        console.log(`   ${item.url}\n`);
      });
    }

    return news;
  } finally {
    await browser.disconnect();
    console.log('Disconnected from Steel session.');
  }
}

scrape().catch(err => { console.error('Error:', err.message); process.exit(1); });
