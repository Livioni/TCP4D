import {test, expect} from '@playwright/test';

test('lock target robots for current and saved runs, and choose new robots in a new run', async ({page}) => {
  const stages = ['track', 'ik', 'mask', 'background', 'inpaint', 'render', 'export'];
  const info = {path: '/episodes/sample', name: 'sample', camera: 'camera', cameras: ['camera'],
    count: 126, width: 320, height: 180, fps: 15, instruction: []};
  const makeJob = (id: string, arms: string[]) => ({id, status: 'ready', created_at: 1,
    episode: info, request: {episode: info.path, camera: info.camera, arms,
      mask_method: 'sam3', query: [160, 90], text: 'robot arm', prompts: []},
    stages: Object.fromEntries(stages.map(stage => [stage, {status: 'pending'}]))});
  const jobs = [makeJob('saved', ['franka_panda'])];
  const createdArms: string[][] = [];
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/capabilities') return route.fulfill({json: {
      arms: ['ur5_wsg', 'franka_panda'], stages, default_mask: 'sam3', data_roots: [],
      checks: ['ur5_wsg', 'franka_panda'].map(arm => ({name: 'asset:' + arm, ok: true, detail: ''})),
    }});
    if (path === '/api/episodes') return route.fulfill({json: [{path: info.path, name: info.name}]});
    if (path === '/api/episodes/inspect') return route.fulfill({json: info});
    if (path === '/api/episodes/frame') return route.fulfill({status: 204});
    if (path === '/api/jobs') {
      if (route.request().method() === 'POST') {
        const {arms} = route.request().postDataJSON();
        createdArms.push(arms);
        const job = makeJob('created-' + createdArms.length, arms);
        jobs.unshift(job);
        return route.fulfill({status: 201, json: job});
      }
      return route.fulfill({json: jobs});
    }
    if (path.endsWith('/events')) return route.fulfill({contentType: 'text/event-stream', body: ': connected\n\n'});
    if (path.endsWith('/artifacts')) return route.fulfill({json: []});
    return route.fulfill({status: 404});
  });

  await page.goto('/');
  await page.getByRole('button', {name: 'Inspect input', exact: true}).click();
  const ur5 = page.getByRole('checkbox', {name: 'ur5 wsg', exact: true});
  const franka = page.getByRole('checkbox', {name: 'franka panda', exact: true});
  await expect(ur5).toBeEnabled();
  await franka.uncheck();
  await page.getByRole('button', {name: 'Create run', exact: true}).click();
  await page.getByText('Source & robots', {exact: true}).click();
  await expect(ur5).toBeChecked();
  await expect(franka).not.toBeChecked();
  await expect(ur5).toBeDisabled();
  await expect(franka).toBeDisabled();
  await expect(page.getByText('Target robots · Locked', {exact: true})).toBeVisible();

  await page.getByRole('button', {name: 'runs', exact: true}).click();
  await page.locator('.run-row').filter({hasText: 'franka_panda'}).click();
  await page.getByText('Source & robots', {exact: true}).click();
  await expect(franka).toBeChecked();
  await expect(ur5).not.toBeChecked();
  await expect(franka).toBeDisabled();
  await expect(ur5).toBeDisabled();

  await page.getByRole('button', {name: 'New run', exact: true}).click();
  await expect(ur5).toBeEnabled();
  await expect(franka).toBeEnabled();
  await ur5.check();
  await franka.uncheck();
  await page.getByRole('button', {name: 'Create run', exact: true}).click();
  await page.getByText('Source & robots', {exact: true}).click();
  await expect(ur5).toBeDisabled();
  await expect(ur5).toBeChecked();
  await expect(franka).toBeDisabled();
  await expect(franka).not.toBeChecked();
  expect(createdArms).toEqual([['ur5_wsg'], ['ur5_wsg']]);
  expect(jobs.find(job => job.id === 'saved')!.request.arms).toEqual(['franka_panda']);
});
