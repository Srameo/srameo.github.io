#!/usr/bin/env node

const fs = require('fs');
const https = require('https');
const path = require('path');

const token = process.env.GITHUB_TOKEN;
if (!token) {
  console.error('Error: GITHUB_TOKEN is not defined');
  process.exit(1);
}

const filePath = path.resolve(process.cwd(), 'projects/galaxy.json');
const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

function fetchStars(githubUrl) {
  return new Promise(resolve => {
    try {
      const [owner, repo] = new URL(githubUrl).pathname.slice(1).split('/');
      const options = {
        hostname: 'api.github.com',
        path: `/repos/${owner}/${repo}`,
        headers: {
          'User-Agent': 'fetch-stars-script',
          Authorization: `token ${token}`
        }
      };
      https.get(options, res => {
        let raw = '';
        res.on('data', chunk => raw += chunk);
        res.on('end', () => {
          try {
            const json = JSON.parse(raw);
            resolve(json.stargazers_count || 0);
          } catch {
            resolve(0);
          }
        });
      }).on('error', () => resolve(0));
    } catch {
      resolve(0);
    }
  });
}

(async () => {
  console.log('Fetching GitHub star counts...');
  for (const item of data) {
    if (item.github_url) {
      // 有 URL 则取 star，没有则置 0
      // eslint-disable-next-line no-await-in-loop
      item.stars = await fetchStars(item.github_url);
    } else {
      item.stars = 0;
    }
  }

  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
  console.log('✅ galaxy.json updated with star counts');
})();
