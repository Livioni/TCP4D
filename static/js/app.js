import { media } from './media.js';
import { enhanceVideo } from './player.js';

const entries = Object.keys(media);
const directory = (fragment) => {
  const path = entries.find((path) => path.includes(fragment));
  if (!path) throw new Error(`Missing video directory: ${fragment}`);
  return path.slice(0, path.lastIndexOf('/'));
};
const el = (tag, className = '', content = '') => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content) node.textContent = content;
  return node;
};
const createButton = (text, className, action) => {
  const button = el('button', className, text);
  button.type = 'button';
  button.addEventListener('click', action);
  return button;
};
const stopVideos = (root) => {
  root.querySelectorAll('video').forEach((video) => {
    video.pause();
    video.removeAttribute('src');
    video.querySelectorAll('source').forEach((source) => source.removeAttribute('src'));
    video.load();
  });
};
function choices(root, label, options, selected, onChange) {
  const bar = el('div', 'selector-bar');
  const labelNode = el('span', 'selector-label', label);
  const group = el('div', 'choice-row');
  group.setAttribute('role', 'group');
  group.setAttribute('aria-label', label);
  const buttons = options.map((option) => {
    const button = createButton(option.label, 'choice', () => {
      selected = option.id;
      buttons.forEach((item, index) => item.setAttribute('aria-pressed', String(options[index].id === selected)));
      onChange(option.id);
    });
    button.setAttribute('aria-pressed', String(option.id === selected));
    group.append(button);
    return button;
  });
  group.addEventListener('keydown', (event) => {
    const index = buttons.indexOf(document.activeElement);
    if (index < 0) return;
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % buttons.length;
    if (event.key === 'ArrowLeft') next = (index - 1 + buttons.length) % buttons.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = buttons.length - 1;
    if (next === undefined) return;
    event.preventDefault();
    buttons[next].focus();
    buttons[next].click();
  });
  bar.append(labelNode, group);
  root.append(bar);
  return bar;
}

function renderVideos(root, { title, meta = '', videos }) {
  stopVideos(root);
  root.replaceChildren();
  const titleRow = el('div', 'panel-title');
  titleRow.append(el('h3', '', title));
  if (meta && meta !== title && !title.startsWith(`${meta} ·`)) titleRow.append(el('span', '', meta));
  const grid = el('div', `video-grid${videos.length === 1 ? ' single' : videos.length === 3 ? ' three' : ''}`);
  const players = [];
  const status = el('p', 'media-status');
  status.setAttribute('role', 'status');
  const failedPlayers = new Set();
  for (const item of videos) {
    if (!media[item.path]) throw new Error(`Missing video: ${item.path}`);
    const asset = media[item.path];
    const figure = el('figure', `video-tile${item.source ? ' source' : ''}`);
    const label = el('figcaption', 'video-label', item.label);
    const frame = el('div', 'video-frame');
    const video = el('video');
    video.controls = true;
    video.playsInline = true;
    video.preload = 'none';
    video.muted = true;
    video.poster = asset.poster;
    video.width = asset.width;
    video.height = asset.height;
    if (asset.fallbackSrc) {
      // Prefer the hosted HEVC original; browsers without HEVC use its current H.264 copy.
      for (const [src, type] of [[asset.src, 'video/mp4; codecs="hvc1"'], [asset.fallbackSrc, 'video/mp4']]) {
        const source = el('source');
        source.src = src;
        source.type = type;
        video.append(source);
      }
    } else video.src = asset.src;
    video.setAttribute('aria-label', `${title}: ${item.label}`);
    video.addEventListener('error', () => {
      // Resetting an old panel can abort its pending request; detached videos need no message.
      if (!root.contains(video) || (!video.getAttribute('src') && !video.querySelector('source[src]'))) return;
      failedPlayers.add(video);
      status.textContent = 'A video could not load. Use its “Open video” link to watch or download the original.';
    });
    const link = el('a', 'video-link', '↗');
    link.title = 'Open original video';
    link.href = asset.originalSrc || asset.src;
    link.target = '_blank';
    link.rel = 'noopener';
    link.setAttribute('aria-label', `Open original video: ${title}, ${item.label}`);
    frame.append(video);
    enhanceVideo(video, frame, () => {
      status.textContent = 'The video could not start. Use its “Open video” link to watch the original.';
    });
    figure.append(label, frame, link);
    grid.append(figure);
    players.push(video);
  }
  const bar = el('div', 'playback-bar');
  let pending = false;
  const update = () => {
    play.textContent = players.some((video) => !video.paused) ? 'Ⅱ Pause all' : '▷ Play all';
  };
  const play = createButton('▷ Play all', 'play-button', async () => {
    if (pending) return;
    if (players.some((video) => !video.paused)) {
      players.forEach((video) => video.pause());
      update();
      return;
    }
    pending = true;
    play.disabled = true;
    play.textContent = 'Loading…';
    // All start requests originate in the same click, preserving mobile playback permission.
    const results = await Promise.allSettled(players.map((video) => video.play()));
    if (root.contains(play)) {
      if (results.some((result) => result.status === 'rejected')) {
        status.textContent = 'Some videos could not start together. Try the individual play controls or open the original video.';
      } else if (!failedPlayers.size) {
        status.textContent = '';
      }
      play.disabled = false;
      update();
    }
    pending = false;
  });
  const restart = createButton('↺ Restart', 'text-button', () => {
    players.forEach((video) => { if (video.readyState > 0) video.currentTime = 0; });
  });
  players.forEach((video) => {
    video.addEventListener('play', update);
    video.addEventListener('pause', update);
    video.addEventListener('ended', update);
  });
  bar.append(play, restart);
  root.append(titleRow, grid, bar, status);
}

const trackingScenes = [
  { id: 'penn', label: 'DROID · PennPAL', dir: directory('S1_TCP4D_Tracking/PennPAL'), cameras: ['25455306', '27085680'], dataset: 'DROID' },
  { id: 'tri', label: 'DROID · TRI', dir: directory('S1_TCP4D_Tracking/TRI'), cameras: ['28451778', '28813166'], dataset: 'DROID' },
  { id: 'iprl', label: 'DROID · IPRL', dir: directory('S1_TCP4D_Tracking/IPRL'), cameras: ['21582473', '28221883'], dataset: 'DROID' },
  { id: 'food-packing', label: 'RoboLab · Food packing', dir: directory('S1_TCP4D_Tracking/FoodPacking1Cans'), cameras: ['third_person'], dataset: 'RoboLab · unseen' },
  { id: 'mustard', label: 'RoboLab · Mustard above raisin', dir: directory('S1_TCP4D_Tracking/MustardAboveRaisin'), cameras: ['third_person'], dataset: 'RoboLab · unseen' },
  { id: 'bottles', label: 'RoboLab · Sauce bottles', dir: directory('S1_TCP4D_Tracking/SauceBottles'), cameras: ['third_person'], dataset: 'RoboLab · unseen' },
];
function trackingExplorer() {
  const root = document.getElementById('tracking-explorer');
  let scene = trackingScenes[0];
  let camera = scene.cameras[0];
  const render = () => renderVideos(body, {
    title: scene.label,
    meta: scene.dataset,
    videos: [
      { label: 'Camera view', path: `${scene.dir}/${camera}.mp4`, source: true },
      { label: 'Predicted point cloud', path: `${scene.dir}/${camera}_pcd.mp4` },
    ],
  });
  choices(root, 'Scene', trackingScenes, scene.id, (id) => {
    scene = trackingScenes.find((item) => item.id === id);
    camera = scene.cameras[0];
    updateCameras();
    render();
  });
  const cameraBar = el('div', 'selector-bar');
  const cameraLabel = el('label', '', 'Camera');
  cameraLabel.htmlFor = 'tracking-camera';
  const select = el('select');
  select.id = 'tracking-camera';
  const updateCameras = () => {
    select.replaceChildren(...scene.cameras.map((value) => {
      const option = el('option', '', value === 'third_person' ? 'Third-person view' : `Camera ${value}`);
      option.value = value;
      return option;
    }));
  };
  select.addEventListener('change', () => { camera = select.value; render(); });
  cameraBar.append(cameraLabel, select);
  root.append(cameraBar);
  const body = el('div', 'explorer-body');
  root.append(body);
  updateCameras();
  render();
}

const augmentationScenes = [
  { id: 'droid-cables', dataset: 'DROID', label: 'Remove cables', dir: directory('S2_augumented_data/DROID/03_Sat'), source: 'original.mp4' },
  { id: 'droid-autolab', dataset: 'DROID', label: 'AUTOLab', dir: directory('S2_augumented_data/DROID/AUTOLab'), source: 'original.mp4' },
  { id: 'robotwin', dataset: 'RoboTwin', label: 'Aloha-AgileX transfer', dir: directory('S2_augumented_data/RoboTwin/'), source: 'source_demo_aloha-agilex.mp4' },
  { id: 'robolab-banana', dataset: 'RoboLab', label: 'Banana on plate', dir: directory('S2_augumented_data/Robolab/Banana'), source: 'original.mp4' },
  { id: 'robolab-bowls', dataset: 'RoboLab', label: 'Stack bowls', dir: directory('S2_augumented_data/Robolab/Bowl'), source: 'original.mp4' },
];
const robotName = (path) => {
  if (path.includes('franka_panda')) return 'Franka Panda';
  if (path.includes('jaco')) return 'Jaco';
  if (path.includes('ur5')) return 'UR5-WSG';
  if (path.includes('arx')) return 'ARX-X5';
  if (path.includes('piper')) return 'Piper';
  return path.split('/').pop().replace('.mp4', '');
};
function augmentationExplorer() {
  const root = document.getElementById('augmentation-explorer');
  let dataset = 'DROID';
  let scene = augmentationScenes[0];
  const targets = () => entries.filter((path) => path.startsWith(`${scene.dir}/`) && path !== `${scene.dir}/${scene.source}`);
  let target = targets()[0];
  const render = () => renderVideos(body, {
    title: `${scene.dataset} · ${scene.label}`,
    meta: `Source → ${robotName(target)}`,
    videos: [
      { label: scene.dataset === 'RoboTwin' ? 'Source · Aloha-AgileX' : 'Source demonstration', path: `${scene.dir}/${scene.source}`, source: true },
      { label: `Target · ${robotName(target)}`, path: target },
    ],
  });
  const refreshTargets = () => {
    targetBar.replaceChildren();
    choices(targetBar, 'Target', targets().map((path) => ({ id: path, label: robotName(path) })), target, (id) => { target = id; render(); });
  };
  const refreshScenes = () => {
    sceneBar.replaceChildren();
    choices(sceneBar, 'Example', augmentationScenes.filter((item) => item.dataset === dataset), scene.id, (id) => {
      scene = augmentationScenes.find((item) => item.id === id);
      target = targets()[0];
      refreshTargets();
      render();
    });
  };
  choices(root, 'Dataset', ['DROID', 'RoboTwin', 'RoboLab'].map((id) => ({ id, label: id })), dataset, (id) => {
    dataset = id;
    scene = augmentationScenes.find((item) => item.dataset === dataset);
    target = targets()[0];
    refreshScenes();
    refreshTargets();
    render();
  });
  const sceneBar = el('div');
  const targetBar = el('div');
  const body = el('div', 'explorer-body');
  root.append(sceneBar, targetBar, body);
  refreshScenes();
  refreshTargets();
  render();
}

function robotExplorer() {
  const root = document.getElementById('robot-explorer');
  let task = 'task1';
  let variant = 'fast';
  const render = () => renderVideos(body, {
    title: task === 'task1' ? 'Placing pens' : 'Stacking bowls',
    meta: variant === 'fast' ? 'cuRobo · CUDA-accelerated IK' : 'SciPy · joint-motion penalty',
    videos: [{ label: 'TCP4D · real robot rollout', path: `${task}${variant === 'long' ? '_long' : ''}_handbrake.mp4` }],
  });
  choices(root, 'Task', [{ id: 'task1', label: 'Placing pens' }, { id: 'task2', label: 'Stacking bowls' }], task, (id) => { task = id; render(); });
  choices(root, 'Variant', [{ id: 'fast', label: 'Fast · cuRobo' }, { id: 'long', label: 'Long · motion penalty' }], variant, (id) => { variant = id; render(); });
  const body = el('div', 'explorer-body');
  root.append(body);
  render();
}

function adaptationExplorer() {
  const root = document.getElementById('adaptation-explorer');
  let episode = '100';
  const render = () => renderVideos(body, {
    title: `UR5 · Episode ${episode}`,
    meta: '10-shot adaptation',
    videos: [
      { label: '3D visualization · third-person', path: `S4_few_shot_adaptation/episode ${episode}/UR5_few_shot_third_person_${episode}.mov`, source: true },
      { label: '3D visualization · alternate view', path: `S4_few_shot_adaptation/episode ${episode}/ur5_${episode}.mov` },
    ],
  });
  choices(root, 'Episode', ['100', '303'].map((id) => ({ id, label: `Episode ${id}` })), episode, (id) => { episode = id; render(); });
  const body = el('div', 'explorer-body');
  root.append(body);
  render();
}

trackingExplorer();
augmentationExplorer();
robotExplorer();
adaptationExplorer();

const hero = document.getElementById('hero-video');
enhanceVideo(hero, hero.parentElement, () => {
  const caption = document.querySelector('.hero-demo .media-caption');
  if (!caption.querySelector('.hero-error')) {
    caption.prepend(el('span', 'hero-error', 'The embedded video could not start. '));
  }
});
hero.addEventListener('error', () => {
  const caption = document.querySelector('.hero-demo .media-caption');
  if (!caption.querySelector('.hero-error')) {
    caption.prepend(el('span', 'hero-error', 'The embedded video could not load. '));
  }
});

document.getElementById('copy-citation').addEventListener('click', async () => {
  const code = document.getElementById('bibtex');
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(code.textContent);
    status.textContent = 'Citation copied.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});

// Track the nearest section without changing the URL or scrolling the page.
const navLinks = [...document.querySelectorAll('.nav-links a')];
const sections = navLinks.map((link) => document.querySelector(link.getAttribute('href')));
const navObserver = new IntersectionObserver((records) => {
  const current = records.filter((record) => record.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
  if (!current) return;
  navLinks.forEach((link) => {
    const active = link.hash === `#${current.target.id}`;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}, { rootMargin: '-90px 0px -65% 0px', threshold: 0 });
sections.forEach((section) => navObserver.observe(section));
navObserver.observe(document.querySelector('.hero'));

document.addEventListener('visibilitychange', () => {
  if (document.hidden) document.querySelectorAll('video').forEach((video) => video.pause());
});
