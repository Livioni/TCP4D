import {defineConfig} from '@playwright/test';
export default defineConfig({
  testDir: './e2e', testMatch: '**/*.e2e.ts', workers: 1, timeout: 180_000,
  use: {baseURL: process.env.AUG_BASE_URL || 'http://127.0.0.1:7862', viewport: {width: 1440, height: 1000},
        launchOptions: {executablePath: process.env.AUG_BROWSER_PATH}},
  outputDir: '../work/browser-results',
});
