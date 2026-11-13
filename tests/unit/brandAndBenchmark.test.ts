import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { runDeviceBenchmark } from '../../src/app/benchmark';
import {
  RULES_BOOK_SECTIONS_FA,
  formatSecondsFa,
  handIndexLabelFa,
  sideLabelFa,
  toPersianDigits,
} from '../../src/app/i18n';
import { renderBrandLogoMarkup } from '../../src/render/brandLogo';

describe('Brand SVG Logo, Persian Typography & Device Benchmark', () => {
  it('renders bespoke self-contained SVG logo and favicon with zero external URLs and unique gradient IDs', () => {
    const heroSvg = renderBrandLogoMarkup({ variant: 'hero', size: 112, idPrefix: 'hero' });
    const topSvg = renderBrandLogoMarkup({ variant: 'compact', size: 46, idPrefix: 'topbar' });
    const faviconSvg = readFileSync(resolve(process.cwd(), 'public/favicon.svg'), 'utf8');

    for (const svg of [heroSvg, topSvg, faviconSvg]) {
      expect(svg).toContain('<svg');
      expect(svg).toContain('viewBox="0 0 128 128"');
      const withoutXmlns = svg.replace('http://www.w3.org/2000/svg', '');
      expect(withoutXmlns).not.toMatch(/https?:\/\//i);
      expect(withoutXmlns).not.toContain('<image');
    }

    expect(heroSvg).toContain('id="hero-gold"');
    expect(topSvg).toContain('id="topbar-gold"');
    expect(heroSvg).toContain('role="img"');
    expect(
      renderBrandLogoMarkup({ decorative: true, idPrefix: 'dec' })
    ).toContain('aria-hidden="true"');
  });

  it('formats Persian digits, hand labels, and complete Persian rules book sections', () => {
    expect(toPersianDigits('2026-1405')).toBe('۲۰۲۶-۱۴۰۵');
    expect(formatSecondsFa(4200)).toBe('۵ ثانیه');
    expect(sideLabelFa('LEFT')).toBe('دست چپ');
    expect(sideLabelFa('RIGHT')).toBe('دست راست');
    expect(handIndexLabelFa(0, 2)).toBe('دست چپ');
    expect(handIndexLabelFa(3, 4)).toBe('دست ۴');
    expect(RULES_BOOK_SECTIONS_FA.length).toBeGreaterThanOrEqual(4);
  });

  it('runs the in-app device benchmark and returns real non-negative timing metrics', async () => {
    const report = await runDeviceBenchmark({ rafFrames: 5, onnxRuns: 5 });

    expect(report.featureExtraction.runs).toBeGreaterThan(0);
    expect(report.featureExtraction.meanMs).toBeGreaterThanOrEqual(0);
    expect(report.hmmFilterStep.runs).toBeGreaterThan(0);
    expect(report.hmmFilterStep.meanMs).toBeGreaterThanOrEqual(0);
    expect(report.onnxInference.runs).toBe(5);
    expect(report.onnxInference.meanMs).toBeGreaterThanOrEqual(0);
    expect(report.estimatedPipelineFrameMs).toBeGreaterThanOrEqual(0);
  });
});
