// Keep playback controls outside the picture so even edge annotations stay visible.
const icons = {
  play: '<path d="m9 5 11 7-11 7Z"/>',
  pause: '<path d="M8 5v14M16 5v14"/>',
  sound: '<path d="m11 5-6 4H2v6h3l6 4ZM15 8a6 6 0 0 1 0 8M18 5a10 10 0 0 1 0 14"/>',
  muted: '<path d="m11 5-6 4H2v6h3l6 4ZM16 9l6 6M22 9l-6 6"/>',
  fullscreen: '<path d="M8 3H3v5M16 3h5v5M21 16v5h-5M8 21H3v-5"/>',
};
const icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${icons[name]}</svg>`;
const time = (seconds) => {
  const value = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, '0')}`;
};

export function enhanceVideo(video, frame, onError) {
  const title = video.getAttribute('aria-label') || 'Video';
  const controls = document.createElement('div');
  controls.className = 'player-controls';
  controls.setAttribute('role', 'group');
  controls.setAttribute('aria-label', `Playback controls: ${title}`);
  const button = (label, symbol, action) => {
    const node = document.createElement('button');
    node.type = 'button';
    node.className = 'player-button';
    node.setAttribute('aria-label', `${label}: ${title}`);
    node.title = label;
    node.innerHTML = icon(symbol);
    node.addEventListener('click', action);
    return node;
  };
  const reportError = (error) => {
    // Scene changes deliberately abort pending playback requests.
    if (error.name !== 'AbortError' && video.isConnected) onError?.();
  };
  const play = button('Play video', 'play', () => {
    if (video.paused) video.play().catch(reportError);
    else video.pause();
  });
  const seek = document.createElement('input');
  seek.type = 'range';
  seek.className = 'player-seek';
  seek.min = '0';
  seek.max = '1000';
  seek.step = '1';
  seek.value = '0';
  seek.disabled = true;
  seek.setAttribute('aria-label', `Seek: ${title}`);
  const clock = document.createElement('span');
  clock.className = 'player-time';
  const sound = button('Mute video', 'sound', () => { video.muted = !video.muted; });
  const fullscreen = button('Enter fullscreen', 'fullscreen', async () => {
    try {
      if (document.fullscreenElement || document.webkitFullscreenElement) {
        if (document.exitFullscreen) await document.exitFullscreen();
        else document.webkitExitFullscreen?.();
      } else if (frame.requestFullscreen) await frame.requestFullscreen();
      else if (frame.webkitRequestFullscreen) frame.webkitRequestFullscreen();
      else video.webkitEnterFullscreen?.();
    } catch (error) { reportError(error); }
  });
  fullscreen.hidden = !(frame.requestFullscreen || frame.webkitRequestFullscreen || video.webkitEnterFullscreen);
  const updateTime = () => {
    const duration = video.duration;
    const ready = Number.isFinite(duration) && duration > 0;
    seek.disabled = !ready;
    seek.value = ready ? String(video.currentTime / duration * 1000) : '0';
    seek.setAttribute('aria-valuetext', `${time(video.currentTime)} of ${time(duration)}`);
    clock.textContent = `${time(video.currentTime)} / ${time(duration)}`;
  };
  const updatePlayback = () => {
    const label = video.paused ? 'Play video' : 'Pause video';
    play.innerHTML = icon(video.paused ? 'play' : 'pause');
    play.setAttribute('aria-label', `${label}: ${title}`);
    play.title = label;
  };
  const updateSound = () => {
    const muted = video.muted || video.volume === 0;
    const label = muted ? 'Unmute video' : 'Mute video';
    sound.innerHTML = icon(muted ? 'muted' : 'sound');
    sound.setAttribute('aria-label', `${label}: ${title}`);
    sound.title = label;
  };
  seek.addEventListener('input', () => {
    if (!Number.isFinite(video.duration) || video.duration <= 0) return;
    const target = Number(seek.value) / 1000 * video.duration;
    video.currentTime = target;
    // Keep the requested position visible while the browser completes its seek.
    clock.textContent = `${time(target)} / ${time(video.duration)}`;
    seek.setAttribute('aria-valuetext', `${time(target)} of ${time(video.duration)}`);
  });
  video.addEventListener('loadedmetadata', () => {
    // Posters reserve space before playback; the actual media dimensions take over once known.
    if (video.videoWidth && video.videoHeight) {
      video.width = video.videoWidth;
      video.height = video.videoHeight;
    }
    updateTime();
  });
  for (const event of ['timeupdate', 'durationchange', 'seeked', 'emptied']) video.addEventListener(event, updateTime);
  for (const event of ['play', 'pause', 'ended']) video.addEventListener(event, updatePlayback);
  video.addEventListener('volumechange', updateSound);
  const updateFullscreen = () => {
    const active = (document.fullscreenElement || document.webkitFullscreenElement) === frame;
    const label = active ? 'Exit fullscreen' : 'Enter fullscreen';
    fullscreen.setAttribute('aria-label', `${label}: ${title}`);
    fullscreen.title = label;
  };
  frame.addEventListener('fullscreenchange', updateFullscreen);
  frame.addEventListener('webkitfullscreenchange', updateFullscreen);
  controls.append(play, seek, clock, sound, fullscreen);
  frame.append(controls);
  // Native controls remain available when JavaScript is disabled or initialization fails.
  video.controls = false;
  updateTime();
  updatePlayback();
  updateSound();
}
