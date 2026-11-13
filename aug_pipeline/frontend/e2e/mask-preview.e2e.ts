import {test, expect, type Page} from '@playwright/test';

async function workbench(page: Page, savedPrompts: any[] = [], tracked = true, query: number[] | null = [195.1, 58.2]) {
  const info = {path: '/samples/episode', name: 'RGB sample', camera: 'cam', cameras: ['cam'], count: 126,
    width: 320, height: 180, fps: 15, instruction: ['Move the object']};
  const stages = ['track', 'ik', 'mask', 'background', 'inpaint', 'render', 'export'];
  let job: any = {id: 'a'.repeat(32), status: 'ready', created_at: 1, episode: info,
    request: {episode: info.path, camera: 'cam', arms: ['ur5_wsg', 'franka_panda'], mask_method: 'sam3',
      query, text: 'robot arm', prompts: savedPrompts},
    stages: Object.fromEntries(stages.map(s => [s, {status: s === 'track' && tracked ? 'complete' : 'pending'}]))};
  const requests: {type: string; body: any}[] = [];
  let created = savedPrompts.length > 0;
  let images: Record<string, string> = {};
  let preview = async (_body: any, _index: number) => images.green;
  let previewCount = 0;
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    const method = route.request().method();
    const json = (value: unknown) => route.fulfill({json: value});
    if (path === '/api/capabilities') return json({arms: ['ur5_wsg', 'franka_panda'], stages, default_mask: 'sam3',
      checks: ['sam3', 'sam3_checkpoint', 'asset:ur5_wsg', 'asset:franka_panda'].map(name => ({name, ok: true}))});
    if (path === '/api/episodes') return json([{path: info.path, name: info.name}]);
    if (path.endsWith('/episodes/inspect')) return json(info);
    if (path.endsWith('/episodes/frame')) return route.fulfill({contentType: 'image/png', body: Buffer.from(images.gray.split(',')[1], 'base64')});
    if (path === '/api/jobs') {
      if (method === 'POST') {created = true; return json(job);}
      return json(created ? [job] : []);
    }
    if (path === `/api/jobs/${job.id}` && method === 'PATCH') {
      const body = route.request().postDataJSON();
      requests.push({type: 'query', body});
      job = {...job, status: 'ready', request: {...job.request, ...body},
        stages: Object.fromEntries(stages.map(s => [s, {status: 'pending'}]))};
      return json(job);
    }
    if (path.endsWith('/events')) return route.fulfill({contentType: 'text/event-stream',
      body: `retry: 600000\nevent: status\ndata: ${JSON.stringify(job)}\n\n`});
    if (path.endsWith('/artifacts')) return json([]);
    if (path.endsWith('/mask-preview')) {
      const body = route.request().postDataJSON();
      const index = previewCount++;
      requests.push({type: 'preview', body});
      try {return json({frame: body.frame, image: await preview(body, index)});}
      catch {return route.fulfill({status: 400, json: {detail: 'Preview unavailable'}});}
    }
    if (path.endsWith('/prompts')) {
      const body = route.request().postDataJSON();
      requests.push({type: 'save', body});
      job = {...job, request: {...job.request, prompts: body.prompts.map((p: any) => ({...p, box: p.box ?? null}))}};
      return json(job);
    }
    if (path.endsWith('/run')) {
      const body = route.request().postDataJSON();
      requests.push({type: 'run', body});
      job = {...job, status: 'ready', stages: {...job.stages, [body.stage]: {status: 'complete'}}};
      return json(job);
    }
    return json(job);
  });
  images = await page.evaluate(() => Object.fromEntries(['gray', 'red', 'green'].map(color => {
    const canvas = document.createElement('canvas'); canvas.width = 320; canvas.height = 180;
    const context = canvas.getContext('2d')!;
    context.fillStyle = {gray: '#888888', red: '#ff0000', green: '#00ff00'}[color]!;
    context.fillRect(0, 0, 320, 180);
    return [color, canvas.toDataURL()];
  })));
  await page.goto('/');
  if (savedPrompts.length) {
    await page.getByRole('button', {name: 'runs', exact: true}).click();
    await page.locator('.run-row').click();
  } else {
    await page.getByRole('button', {name: 'Inspect input', exact: true}).click();
    await page.getByRole('button', {name: 'Create run', exact: true}).click();
  }
  await page.getByRole('button', {name: 'Mask', exact: true}).click();
  const canvas = page.getByLabel('Mask correction image');
  const click = async (x: number, y: number) => {
    const bounds = (await canvas.boundingBox())!;
    await canvas.click({position: {x: x / 320 * bounds.width, y: y / 180 * bounds.height}});
  };
  const pixel = () => canvas.evaluate(element => Array.from((element as HTMLCanvasElement).getContext('2d')!.getImageData(0, 0, 1, 1).data).slice(0, 3));
  return {requests, images, click, pixel, setPreview: (handler: typeof preview) => {preview = handler;}};
}

test('points update only the current mask; propagation is a separate explicit action', async ({page}) => {
  const app = await workbench(page);
  const releases: (() => void)[] = [];
  app.setPreview(async (_body, index) => {
    await new Promise<void>(resolve => {releases[index] = resolve;});
    return index === 0 ? app.images.red : app.images.green;
  });
  const propagate = page.getByRole('button', {name: 'Propagate', exact: true});
  await app.click(260, 100);
  await expect.poll(() => app.requests.length).toBe(1);
  await expect(page.getByRole('status')).toHaveText('Updating current mask…');
  await expect(propagate).toBeDisabled();
  await page.getByRole('button', {name: '− Background', exact: true}).click();
  await app.click(100, 130);
  expect(app.requests.map(r => r.type)).toEqual(['preview']);
  releases[0]();
  await expect.poll(() => app.requests.length).toBe(2);
  await expect(propagate).toBeDisabled();
  expect(app.requests[1].body.prompts.flatMap((p: any) => p.labels)).toEqual([1, 0]);
  releases[1]();
  await expect(page.getByRole('status')).toHaveText('Current mask updated.');
  await expect.poll(app.pixel).toEqual([0, 255, 0]);
  await expect(propagate).toBeEnabled();
  expect(app.requests.every(r => r.type === 'preview')).toBe(true);
  await page.getByRole('button', {name: 'Undo', exact: true}).click();
  await expect.poll(app.pixel).toEqual([255, 0, 0]);
  await app.click(100, 130);
  await expect.poll(app.pixel).toEqual([0, 255, 0]);
  expect(app.requests).toHaveLength(2);
  await page.getByRole('slider', {name: 'Video frame'}).fill('1');
  await expect.poll(app.pixel).toEqual([136, 136, 136]);
  await page.getByRole('slider', {name: 'Video frame'}).fill('0');
  await expect.poll(app.pixel).toEqual([0, 255, 0]);
  await propagate.click();
  await expect.poll(() => app.requests.map(r => r.type)).toEqual(['preview', 'preview', 'save', 'run']);
  expect(app.requests[2].body.prompts).toHaveLength(2);
  expect(app.requests[3].body).toEqual({stage: 'mask'});
});

test('a late preview cannot replace another frame, and failed updates can be retried', async ({page}) => {
  const app = await workbench(page);
  let release!: () => void;
  app.setPreview(async () => {await new Promise<void>(resolve => {release = resolve;}); return app.images.green;});
  await app.click(260, 100);
  await expect.poll(() => app.requests.length).toBe(1);
  await page.getByRole('slider', {name: 'Video frame'}).fill('1');
  release();
  await expect.poll(app.pixel).toEqual([136, 136, 136]);
  await expect(page.getByRole('status')).toHaveText('Add foreground or background points to update this frame.');
  app.setPreview(async (_body, index) => {if (index === 1) throw new Error('failed'); return app.images.red;});
  await app.click(250, 100);
  await expect(page.getByRole('status')).toHaveText('Mask update failed.');
  await expect(page.getByRole('button', {name: 'Propagate', exact: true})).toBeDisabled();
  await page.getByRole('button', {name: 'Retry', exact: true}).click();
  await expect(page.getByRole('status')).toHaveText('Current mask updated.');
  await expect.poll(app.pixel).toEqual([255, 0, 0]);
  await expect(page.getByRole('button', {name: 'Propagate', exact: true})).toBeEnabled();
});

test('clearing saved prompts can explicitly rebuild automatic masks', async ({page}) => {
  const app = await workbench(page, [{frame: 0, points: [[260, 100]], labels: [1], box: null}]);
  await page.getByRole('button', {name: 'Clear prompts', exact: true}).click();
  await expect(page.getByRole('status')).toHaveText('Prompts cleared. Propagate to rebuild masks.');
  await page.getByRole('button', {name: 'Propagate', exact: true}).click();
  await expect.poll(() => app.requests.map(r => r.type)).toEqual(['save', 'run']);
  expect(app.requests[0].body.prompts).toEqual([]);
});

test('steps require an explicit click and prerequisites never start automatically', async ({page}) => {
  const app = await workbench(page, [], false);
  await expect(page.getByRole('button', {name: /Run pipeline/})).toHaveCount(0);
  await expect(page.getByRole('button', {name: 'Run step', exact: true})).toBeDisabled();
  await expect(page.getByText('Complete Track TCP first, then run this step manually.')).toBeVisible();
  expect(app.requests).toEqual([]);
  await app.click(260, 100);
  await expect(page.getByRole('status')).toHaveText('Current mask updated.');
  await expect(page.getByRole('button', {name: 'Propagate', exact: true})).toBeDisabled();
  await page.getByRole('button', {name: 'Track TCP', exact: true}).click();
  expect(app.requests.filter(r => r.type === 'run')).toHaveLength(0);
  await page.getByRole('button', {name: 'Run step', exact: true}).click();
  await expect.poll(() => app.requests.filter(r => r.type === 'run').map(r => r.body.stage)).toEqual(['track']);
  await page.getByRole('button', {name: 'Mask', exact: true}).click();
  await page.getByRole('button', {name: 'Propagate', exact: true}).click();
  await expect.poll(() => app.requests.filter(r => r.type === 'run').map(r => r.body.stage)).toEqual(['track', 'mask']);
});

test('clicking the TCP saves its query immediately and tracking starts without Prepare', async ({page}) => {
  const app = await workbench(page, [], false, null);
  await expect(page.getByRole('button', {name: 'Prepare', exact: true})).toHaveCount(0);
  await expect(page.getByRole('button', {name: 'Save query', exact: true})).toHaveCount(0);
  await page.getByRole('button', {name: 'Track TCP', exact: true}).click();
  const run = page.getByRole('button', {name: 'Run step', exact: true});
  await expect(run).toBeDisabled();
  const canvas = page.getByLabel('Click to select the initial TCP');
  const bounds = (await canvas.boundingBox())!;
  await canvas.click({position: {x: bounds.width / 2, y: bounds.height / 2}});
  await expect.poll(() => app.requests.map(r => r.type)).toEqual(['query']);
  expect(app.requests[0].body.query[0]).toBeCloseTo(160, 1);
  expect(app.requests[0].body.query[1]).toBeCloseTo(90, 1);
  await expect(page.getByText('TCP query: (160.0, 90.0) px', {exact: true})).toBeVisible();
  await expect(run).toBeEnabled();
  await run.click();
  await expect.poll(() => app.requests.map(r => r.type)).toEqual(['query', 'run']);
  expect(app.requests[1].body).toEqual({stage: 'track', force: false});
  await page.reload();
  await page.getByRole('button', {name: 'runs', exact: true}).click();
  await page.locator('.run-row').click();
  await expect(page.getByRole('heading', {name: 'Track TCP', exact: true})).toBeVisible();
  await expect(page.getByText('TCP query: (160.0, 90.0) px', {exact: true})).toBeVisible();
});
