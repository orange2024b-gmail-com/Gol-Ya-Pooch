export interface BrandLogoOptions {
  size?: number;
  variant?: 'compact' | 'hero';
  idPrefix?: string;
  decorative?: boolean;
}

export function renderBrandLogoMarkup(options: BrandLogoOptions = {}): string {
  const size = options.size ?? (options.variant === 'hero' ? 104 : 44);
  const prefix = (options.idPrefix ?? (options.variant === 'hero' ? 'hero' : 'topbar')).replace(
    /[^a-zA-Z0-9_-]/g,
    ''
  );
  const decorative = options.decorative ?? false;
  const a11yAttrs = decorative
    ? 'aria-hidden="true"'
    : 'role="img" aria-label="نشان بازی گل یا پوچ"';

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="${size}" height="${size}" class="brand-logo-svg brand-logo-${options.variant ?? 'compact'}" ${a11yAttrs}>
  <defs>
    <radialGradient id="${prefix}-bg" cx="50%" cy="42%" r="60%">
      <stop offset="0%" stop-color="#15474b" />
      <stop offset="100%" stop-color="#091316" />
    </radialGradient>
    <linearGradient id="${prefix}-gold" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#ffe08a" />
      <stop offset="55%" stop-color="#f5b942" />
      <stop offset="100%" stop-color="#c6760c" />
    </linearGradient>
    <radialGradient id="${prefix}-pearl" cx="35%" cy="32%" r="65%">
      <stop offset="0%" stop-color="#fffbeb" />
      <stop offset="45%" stop-color="#ffd56b" />
      <stop offset="100%" stop-color="#d98218" />
    </radialGradient>
    <radialGradient id="${prefix}-aura" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#ffcf6b" stop-opacity="0.58" />
      <stop offset="100%" stop-color="#f5b942" stop-opacity="0" />
    </radialGradient>
  </defs>
  <circle cx="64" cy="64" r="60" fill="url(#${prefix}-bg)" stroke="url(#${prefix}-gold)" stroke-width="3" />
  <rect x="24" y="24" width="80" height="80" rx="8" fill="none" stroke="url(#${prefix}-gold)" stroke-width="2" stroke-opacity="0.65" />
  <rect x="24" y="24" width="80" height="80" rx="8" transform="rotate(45 64 64)" fill="none" stroke="url(#${prefix}-gold)" stroke-width="2" stroke-opacity="0.65" />
  <circle cx="64" cy="64" r="40" fill="#0d2326" fill-opacity="0.72" stroke="#f5b942" stroke-width="1.2" stroke-opacity="0.35" />
  <path d="M29 66 C30 84, 44 96, 61 97 C54 88, 48 80, 46 69 C45 63, 40 58, 35 60 C31 61, 29 63, 29 66 Z" fill="#115e59" stroke="url(#${prefix}-gold)" stroke-width="2.4" stroke-linejoin="round" />
  <path d="M99 66 C98 84, 84 96, 67 97 C74 88, 80 80, 82 69 C83 63, 88 58, 93 60 C97 61, 99 63, 99 66 Z" fill="#115e59" stroke="url(#${prefix}-gold)" stroke-width="2.4" stroke-linejoin="round" />
  <path d="M40 93 Q64 103 88 93" fill="none" stroke="url(#${prefix}-gold)" stroke-width="2.4" stroke-linecap="round" />
  <circle cx="64" cy="56" r="24" fill="url(#${prefix}-aura)" />
  <circle cx="64" cy="56" r="13.5" fill="url(#${prefix}-pearl)" stroke="#fff7d6" stroke-width="1.8" />
  <circle cx="59.5" cy="51.5" r="4" fill="#ffffff" fill-opacity="0.85" />
  <polygon points="64,22 68,28 64,34 60,28" fill="url(#${prefix}-gold)" />
</svg>`;
}

export function createBrandLogoElement(options: BrandLogoOptions = {}): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = `brand-logo-wrap ${options.variant === 'hero' ? 'is-hero' : 'is-compact'}`;
  wrap.innerHTML = renderBrandLogoMarkup(options);
  return wrap;
}
