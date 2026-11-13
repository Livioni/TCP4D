import {describe, expect, it} from 'vitest';
import {imagePoint, boxFrom} from './geometry';
describe('native image coordinates', () => {
  it('maps scaled display clicks back to the unpadded source', () => {
    expect(imagePoint(420, 230, {left: 100, top: 50, width: 640, height: 360}, 320, 180)).toEqual([160, 90]);
  });
  it('clamps edge clicks to the final pixel', () => {
    expect(imagePoint(800, -20, {left: 0, top: 0, width: 640, height: 360}, 320, 180)).toEqual([319, 0]);
  });
  it('normalizes backward box drags and rejects accidental clicks', () => {
    expect(boxFrom([80, 60], [20, 10])).toEqual([20, 10, 80, 60]);
    expect(boxFrom([20, 10], [21, 11])).toBeNull();
  });
});
