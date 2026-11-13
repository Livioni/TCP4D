import {test, expect} from '@playwright/test';

test('load the default RGB example and select the initial TCP', async ({page, request}) => {
  const inputs = await (await request.get('/api/episodes')).json();
  const info = await (await request.post('/api/episodes/inspect', {data: {episode: inputs[0].path}})).json();
  const errors: string[] = [];
  page.on('pageerror', e=>errors.push(e.message));
  await page.goto('/');
  await page.getByRole('button', {name:'Inspect input',exact:true}).click();
  await expect(page.getByText(`${info.count} frames · ${info.fps} fps`, {exact:true})).toBeVisible();
  const canvas = page.getByLabel('Click to select the initial TCP');
  const bounds = (await canvas.boundingBox())!;
  await canvas.click({position:{x:bounds.width/2,y:bounds.height/2}});
  await expect(page.getByText('TCP query: (160.0, 90.0) px', {exact:true})).toBeVisible();
  await expect(page.getByRole('button', {name:'Use first-frame GT',exact:true})).toHaveCount(0);
  await page.getByRole('button', {name:'resources',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Resources',exact:true})).toBeVisible();
  expect(errors).toEqual([]);
});

test('refine a real SAM3 run, invalidate downstream stages and regenerate paired data', async ({page,request}) => {
  const method = 'sam3';
  const jobs = await (await request.get('/api/jobs')).json();
  const job = jobs.find((j:any)=>j.request.mask_method===method && j.stages.ik.status==='complete');
  test.skip(!job, `Run the full ${method} DROID sample before this GPU integration test`);
  await expect.poll(async()=> (await (await request.get(`/api/jobs/${job.id}`)).json()).status,
                    {timeout:90_000}).not.toMatch(/running|queued/);
  const before = job.stages;
  await page.goto('/');
  await page.getByRole('button',{name:'runs',exact:true}).click();
  await page.locator('.run-row').filter({hasText:method.toUpperCase()}).first().click();
  await page.getByRole('button',{name:'Mask',exact:true}).click();
  await page.getByRole('button',{name:'Clear prompts',exact:true}).click();
  const canvas = page.getByLabel('Mask correction image');
  const bounds = (await canvas.boundingBox())!;
  const px=(x:number,y:number)=>({x:x/320*bounds.width,y:y/180*bounds.height});
  const previousX = job.request.prompts?.[0]?.points?.[0]?.[0];
  const positiveX = previousX >= 260 && previousX < 270 ? previousX + 1 : 260;
  await page.getByRole('button',{name:'+ Foreground',exact:true}).click();
  await canvas.click({position:px(positiveX,100)});
  await page.getByRole('button',{name:'− Background',exact:true}).click();
  await canvas.click({position:px(100,130)});
  await page.getByRole('button',{name:'Box',exact:true}).click();
  await canvas.scrollIntoViewIfNeeded();
  const boxBounds = (await canvas.boundingBox())!;
  await page.mouse.move(boxBounds.x+px(128,1).x,boxBounds.y+px(128,1).y);
  await page.mouse.down();
  await page.mouse.move(boxBounds.x+px(318,172).x,boxBounds.y+px(318,172).y,{steps:8});
  await page.mouse.up();
  await expect(page.getByRole('status')).toHaveText('Current mask updated.', {timeout:90_000});
  const beforePropagation = await (await request.get(`/api/jobs/${job.id}`)).json();
  expect(beforePropagation.request.prompts).toEqual(job.request.prompts);
  expect(beforePropagation.stages).toEqual(before);
  const saved = page.waitForResponse(r=>r.url().endsWith(`/jobs/${job.id}/prompts`) && r.request().method()==='PUT');
  const started = page.waitForResponse(r=>r.url().endsWith(`/jobs/${job.id}/run`) && r.request().method()==='POST');
  await page.getByRole('button',{name:'Propagate',exact:true}).click();
  expect((await saved).ok()).toBe(true);
  expect((await started).ok()).toBe(true);
  await expect.poll(async()=> (await (await request.get(`/api/jobs/${job.id}`)).json()).stages.mask.status,
                    {timeout:90_000}).toMatch(/complete|failed/);
  const corrected = await (await request.get(`/api/jobs/${job.id}`)).json();
  expect(corrected.stages.mask.status).toBe('complete');
  const coverage = await (await request.get(`/api/jobs/${job.id}/artifacts/mask_stats.json`)).json();
  expect(coverage.frames).toBe(126);
  expect(coverage.empty_frames).toHaveLength(0);
  expect(corrected.request.prompts).toHaveLength(3);
  expect(corrected.stages.track.started_at).toBe(before.track.started_at);
  expect(corrected.stages.ik.started_at).toBe(before.ik.started_at);
  expect(corrected.stages.inpaint.status).toBe('pending');
  for (const [stage, name] of [['background', 'Erase & prefill'], ['inpaint', 'Inpaint'], ['render', 'Render'], ['export', 'Export']]) {
    await page.getByRole('button', {name, exact:true}).click();
    const rerun = page.waitForResponse(r=>r.url().endsWith(`/jobs/${job.id}/run`) && r.request().method()==='POST');
    await page.getByRole('button',{name:'Run step',exact:true}).click();
    expect((await rerun).ok()).toBe(true);
    await expect.poll(async()=> (await (await request.get(`/api/jobs/${job.id}`)).json()).stages[stage].status,
                      {timeout:90_000}).toBe('complete');
  }
  const result = await (await request.get(`/api/jobs/${job.id}/artifacts`)).json();
  expect(result.some((a:any)=>a.name==='dataset.zip')).toBe(true);
  expect(result.filter((a:any)=>a.name.startsWith('targets/')&&a.name.endsWith('.mp4'))).toHaveLength(2);
  await page.screenshot({path:`../work/workbench-refined-${method}.png`,fullPage:true});
});
