/**
 * Aurora Night Sky — Dynamic Hover Effects Engine
 * Each interactive element type gets a unique hover reaction.
 *
 * Effects:
 *  1. Cursor glow trail          — teal orb follows mouse everywhere
 *  2. Ripple wave                — click/hover ripple on ALL buttons
 *  3. Magnetic pull              — primary buttons attract to cursor
 *  4. 3D perspective tilt        — stat cards & feature cards tilt toward cursor
 *  5. Aurora border trace        — animated gradient border sweeps around cards on hover
 *  6. Star particle burst        — stat icon emits mini stars on hover
 *  7. Scan line sweep            — emergency rows & nav items get a light scan
 *  8. Shimmer sweep              — secondary buttons get a light shimmer
 *  9. Energy pulse ring          — example-btn level badges pulse a ring outward
 * 10. Neon text glow             — nav active item text brightens with glow
 */

(function () {
  'use strict';

  /* ════════════════════════════════════════════════════════════
     1. CURSOR GLOW TRAIL
  ════════════════════════════════════════════════════════════ */
  const cursorGlow = document.createElement('div');
  cursorGlow.id = 'cursorGlow';
  Object.assign(cursorGlow.style, {
    position:      'fixed',
    width:         '360px',
    height:        '360px',
    borderRadius:  '50%',
    background:    'radial-gradient(circle, rgba(0,212,170,0.07) 0%, rgba(26,107,255,0.04) 40%, transparent 70%)',
    pointerEvents: 'none',
    zIndex:        '9999',
    transform:     'translate(-50%,-50%)',
    transition:    'opacity 0.3s',
    opacity:       '0',
  });
  document.body.appendChild(cursorGlow);

  let mouseX = 0, mouseY = 0, glowX = 0, glowY = 0;

  document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX; mouseY = e.clientY;
    cursorGlow.style.opacity = '1';
    cursorGlow.style.left = mouseX + 'px';
    cursorGlow.style.top  = mouseY + 'px';
  });
  document.addEventListener('mouseleave', () => {
    cursorGlow.style.opacity = '0';
  });


  /* ════════════════════════════════════════════════════════════
     2. RIPPLE WAVE — on all buttons on mousedown
  ════════════════════════════════════════════════════════════ */
  function addRipple(el, e, color = 'rgba(0,212,170,0.35)') {
    const rect   = el.getBoundingClientRect();
    const size   = Math.max(rect.width, rect.height) * 2;
    const x      = (e.clientX - rect.left) - size / 2;
    const y      = (e.clientY - rect.top)  - size / 2;

    const ripple = document.createElement('span');
    Object.assign(ripple.style, {
      position:     'absolute',
      left:         x + 'px',
      top:          y + 'px',
      width:        size + 'px',
      height:       size + 'px',
      borderRadius: '50%',
      background:   color,
      transform:    'scale(0)',
      animation:    'rippleAnim 0.65s ease-out forwards',
      pointerEvents:'none',
      zIndex:       '10',
    });

    // Ensure parent is positioned
    const pos = getComputedStyle(el).position;
    if (pos === 'static') el.style.position = 'relative';
    el.style.overflow = 'hidden';
    el.appendChild(ripple);
    ripple.addEventListener('animationend', () => ripple.remove());
  }

  // Inject ripple keyframe
  const styleSheet = document.createElement('style');
  styleSheet.textContent = `
    @keyframes rippleAnim {
      to { transform: scale(1); opacity: 0; }
    }
    @keyframes scanLine {
      0%   { left: -100%; opacity: 0; }
      10%  { opacity: 1; }
      90%  { opacity: 1; }
      100% { left: 110%; opacity: 0; }
    }
    @keyframes particleFly {
      0%   { transform: translate(0,0) scale(1); opacity: 1; }
      100% { transform: translate(var(--tx), var(--ty)) scale(0); opacity: 0; }
    }
    @keyframes pulseRing {
      0%   { transform: scale(1);   opacity: 0.8; }
      100% { transform: scale(2.2); opacity: 0; }
    }
    @keyframes shimmerSlide {
      0%   { left: -100%; }
      100% { left: 200%; }
    }
    @keyframes borderTrace {
      0%   { background-position: 0% 50%; }
      50%  { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }
    @keyframes magnetFloat {
      0%, 100% { box-shadow: 0 8px 32px rgba(0,212,170,0.4), 0 0 60px rgba(26,107,255,0.2); }
      50%       { box-shadow: 0 14px 48px rgba(0,212,170,0.6), 0 0 80px rgba(26,107,255,0.35); }
    }

    /* Aurora border trace on cards */
    .aurora-border-card {
      position: relative;
    }
    .aurora-border-card::after {
      content: '';
      position: absolute;
      inset: -1px;
      border-radius: inherit;
      padding: 1px;
      background: linear-gradient(
        var(--angle, 0deg),
        transparent 30%,
        rgba(0,212,170,0.8) 50%,
        transparent 70%
      );
      -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
      -webkit-mask-composite: destination-out;
      mask-composite: exclude;
      opacity: 0;
      transition: opacity 0.3s;
      pointer-events: none;
    }
    .aurora-border-card:hover::after {
      opacity: 1;
      animation: borderRotate 2s linear infinite;
    }
    @property --angle {
      syntax: '<angle>';
      initial-value: 0deg;
      inherits: false;
    }
    @keyframes borderRotate {
      to { --angle: 360deg; }
    }

    /* Scan line overlay */
    .scan-host {
      position: relative;
      overflow: hidden;
    }
    .scan-host .scan-line {
      position: absolute;
      top: 0; bottom: 0;
      width: 60px;
      background: linear-gradient(90deg, transparent, rgba(0,212,170,0.35), transparent);
      pointer-events: none;
      opacity: 0;
      left: -100%;
    }
    .scan-host:hover .scan-line {
      animation: scanLine 0.7s ease-out forwards;
    }

    /* Shimmer overlay for ghost buttons */
    .shimmer-host {
      position: relative;
      overflow: hidden;
    }
    .shimmer-host::before {
      content: '';
      position: absolute;
      top: 0; bottom: 0;
      width: 50%;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
      left: -100%;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s;
    }
    .shimmer-host:hover::before {
      opacity: 1;
      animation: shimmerSlide 0.6s ease-out forwards;
    }

    /* 3D tilt wrapper */
    .tilt-card {
      transform-style: preserve-3d;
      will-change: transform;
    }
    .tilt-card .tilt-glare {
      position: absolute;
      inset: 0;
      border-radius: inherit;
      background: radial-gradient(circle at var(--mx,50%) var(--my,50%),
        rgba(0,212,170,0.12) 0%, transparent 60%);
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.3s;
      z-index: 2;
    }
    .tilt-card:hover .tilt-glare { opacity: 1; }

    /* Primary button magnetic glow */
    .btn-primary:hover {
      animation: magnetFloat 1.5s ease-in-out infinite;
    }

    /* Nav item highlight sweep */
    .nav-item {
      overflow: hidden;
    }
    .nav-item .nav-scan {
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, transparent 0%, rgba(0,212,170,0.1) 50%, transparent 100%);
      left: -100%;
      opacity: 0;
      pointer-events: none;
    }
    .nav-item:hover .nav-scan,
    .nav-item.active .nav-scan {
      animation: scanLine 0.5s ease-out forwards;
    }

    /* Stat icon particle container */
    .stat-icon {
      overflow: visible !important;
    }

    /* Example button pulse ring */
    .example-btn .pulse-ring {
      position: absolute;
      inset: 0;
      border-radius: inherit;
      border: 1.5px solid var(--lvl-color, rgba(0,212,170,0.5));
      opacity: 0;
      pointer-events: none;
    }
    .example-btn:hover .pulse-ring {
      animation: pulseRing 0.7s ease-out forwards;
    }
    .example-btn { position: relative; }

    /* Emerg row energy pulse */
    .emerg-row .emerg-pulse {
      position: absolute;
      inset: 0;
      border-radius: inherit;
      border: 1px solid rgba(0,212,170,0.4);
      opacity: 0;
      pointer-events: none;
    }
    .emerg-row:hover .emerg-pulse {
      animation: pulseRing 0.6s ease-out forwards;
    }

    /* Feature card glow orb that follows mouse */
    .feature-card .fc-orb {
      position: absolute;
      width: 200px; height: 200px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(0,212,170,0.15) 0%, transparent 70%);
      pointer-events: none;
      transform: translate(-50%,-50%);
      transition: opacity 0.3s;
      opacity: 0;
      z-index: 0;
    }
    .feature-card:hover .fc-orb { opacity: 1; }
    .feature-card * { position: relative; z-index: 1; }

    /* Stat card inner shine that follows mouse */
    .stat-card .sc-shine {
      position: absolute;
      width: 180px; height: 180px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(0,212,170,0.12) 0%, transparent 70%);
      pointer-events: none;
      transform: translate(-50%,-50%);
      transition: opacity 0.25s;
      opacity: 0;
    }
    .stat-card:hover .sc-shine { opacity: 1; }
  `;
  document.head.appendChild(styleSheet);


  /* ════════════════════════════════════════════════════════════
     3. MAGNETIC PRIMARY BUTTONS
  ════════════════════════════════════════════════════════════ */
  function initMagnetic(el) {
    const STRENGTH = 0.25;
    el.addEventListener('mousemove', (e) => {
      const rect = el.getBoundingClientRect();
      const cx   = rect.left + rect.width  / 2;
      const cy   = rect.top  + rect.height / 2;
      const dx   = (e.clientX - cx) * STRENGTH;
      const dy   = (e.clientY - cy) * STRENGTH;
      el.style.transform = `translate(${dx}px, ${dy}px) scale(1.04)`;
    });
    el.addEventListener('mouseleave', () => {
      el.style.transform = '';
      el.style.transition = 'transform 0.5s cubic-bezier(0.4,0,0.2,1)';
      setTimeout(() => { el.style.transition = ''; }, 500);
    });
    el.addEventListener('mousedown', (e) => addRipple(el, e, 'rgba(0,212,170,0.4)'));
  }


  /* ════════════════════════════════════════════════════════════
     4. 3D PERSPECTIVE TILT + mouse-tracked inner glow
  ════════════════════════════════════════════════════════════ */
  function initTilt(el, intensity = 8) {
    // Add glare div
    const glare = document.createElement('div');
    glare.className = 'tilt-glare';
    el.classList.add('tilt-card');
    el.style.position = 'relative';
    el.appendChild(glare);

    el.addEventListener('mousemove', (e) => {
      const rect  = el.getBoundingClientRect();
      const xPct  = (e.clientX - rect.left)  / rect.width;
      const yPct  = (e.clientY - rect.top)   / rect.height;
      const rotX  = (0.5 - yPct) * intensity;
      const rotY  = (xPct - 0.5) * intensity;
      el.style.transform    = `perspective(600px) rotateX(${rotX}deg) rotateY(${rotY}deg) scale(1.02)`;
      el.style.transition   = 'transform 0.1s linear';
      glare.style.setProperty('--mx', (xPct * 100) + '%');
      glare.style.setProperty('--my', (yPct * 100) + '%');
    });
    el.addEventListener('mouseleave', () => {
      el.style.transform  = '';
      el.style.transition = 'transform 0.6s cubic-bezier(0.4,0,0.2,1)';
    });
  }


  /* ════════════════════════════════════════════════════════════
     5. AURORA BORDER TRACE — rotating gradient border
  ════════════════════════════════════════════════════════════ */
  function initAuroraBorder(el) {
    el.classList.add('aurora-border-card');
  }


  /* ════════════════════════════════════════════════════════════
     6. STAR PARTICLE BURST from stat icons
  ════════════════════════════════════════════════════════════ */
  const STAR_COLORS = ['#00d4aa','#1a6bff','#a855f7','#00b4d8','#ffffff'];

  function burstParticles(iconEl) {
    const rect = iconEl.getBoundingClientRect();
    const cx   = rect.left + rect.width  / 2;
    const cy   = rect.top  + rect.height / 2;
    const count = 10;

    for (let i = 0; i < count; i++) {
      const p   = document.createElement('div');
      const ang = (360 / count) * i;
      const rad = (Math.random() * 30 + 25);
      const tx  = Math.cos(ang * Math.PI / 180) * rad;
      const ty  = Math.sin(ang * Math.PI / 180) * rad;
      const col = STAR_COLORS[Math.floor(Math.random() * STAR_COLORS.length)];
      const sz  = Math.random() * 4 + 2;

      Object.assign(p.style, {
        position:     'fixed',
        left:         cx + 'px',
        top:          cy + 'px',
        width:        sz + 'px',
        height:       sz + 'px',
        borderRadius: '50%',
        background:   col,
        boxShadow:    `0 0 ${sz*2}px ${col}`,
        pointerEvents:'none',
        zIndex:       '9998',
        setProperty:  null,
        '--tx':       tx + 'px',
        '--ty':       ty + 'px',
        animation:    `particleFly ${0.5 + Math.random()*0.3}s ease-out forwards`,
        transform:    'translate(-50%,-50%)',
      });
      p.style.setProperty('--tx', tx + 'px');
      p.style.setProperty('--ty', ty + 'px');
      document.body.appendChild(p);
      p.addEventListener('animationend', () => p.remove());
    }
  }


  /* ════════════════════════════════════════════════════════════
     7. SCAN LINE — for nav items and emerg rows
  ════════════════════════════════════════════════════════════ */
  function addScanLine(el) {
    el.classList.add('scan-host');
    const line = document.createElement('div');
    line.className = 'scan-line';
    el.appendChild(line);

    el.addEventListener('mouseenter', () => {
      line.style.animation = 'none';
      void line.offsetWidth;
      line.style.animation = '';
    });
  }


  /* ════════════════════════════════════════════════════════════
     8. SHIMMER SWEEP — ghost buttons
  ════════════════════════════════════════════════════════════ */
  function addShimmer(el) {
    el.classList.add('shimmer-host');
    el.addEventListener('mousedown', (e) => addRipple(el, e, 'rgba(255,255,255,0.1)'));
  }


  /* ════════════════════════════════════════════════════════════
     9. EXAMPLE BTN PULSE RING
  ════════════════════════════════════════════════════════════ */
  function addPulseRing(el) {
    const ring = document.createElement('div');
    ring.className = 'pulse-ring';
    el.appendChild(ring);

    el.addEventListener('mouseenter', () => {
      ring.style.animation = 'none';
      void ring.offsetWidth;
      ring.style.animation = '';
    });
    el.addEventListener('mousedown', (e) => addRipple(el, e, 'rgba(0,212,170,0.25)'));
  }


  /* ════════════════════════════════════════════════════════════
     10. FEATURE CARD MOUSE-TRACKING ORB
  ════════════════════════════════════════════════════════════ */
  function addFeatureOrb(el) {
    const orb = document.createElement('div');
    orb.className = 'fc-orb';
    el.style.position = 'relative';
    el.style.overflow  = 'hidden';
    el.appendChild(orb);

    el.addEventListener('mousemove', (e) => {
      const rect = el.getBoundingClientRect();
      orb.style.left = (e.clientX - rect.left) + 'px';
      orb.style.top  = (e.clientY - rect.top)  + 'px';
    });
  }


  /* ════════════════════════════════════════════════════════════
     STAT CARD — mouse-tracked shine + 3D tilt + particle burst
  ════════════════════════════════════════════════════════════ */
  function initStatCard(el) {
    // Add shine layer
    const shine = document.createElement('div');
    shine.className = 'sc-shine';
    el.style.position = 'relative';
    el.style.overflow = 'hidden';
    el.appendChild(shine);

    el.addEventListener('mousemove', (e) => {
      const rect = el.getBoundingClientRect();
      shine.style.left = (e.clientX - rect.left) + 'px';
      shine.style.top  = (e.clientY - rect.top)  + 'px';
    });

    // 3D tilt
    initTilt(el, 6);

    // Particle burst from icon on hover
    const icon = el.querySelector('.stat-icon');
    if (icon) {
      el.addEventListener('mouseenter', () => burstParticles(icon));
    }
  }


  /* ════════════════════════════════════════════════════════════
     EMERG ROW — scan + pulse ring + ripple
  ════════════════════════════════════════════════════════════ */
  function initEmergRow(el) {
    addScanLine(el);
    const ring = document.createElement('div');
    ring.className = 'emerg-pulse';
    el.appendChild(ring);

    el.addEventListener('mouseenter', () => {
      ring.style.animation = 'none';
      void ring.offsetWidth;
      ring.style.animation = '';
    });
    el.addEventListener('mousedown', (e) => addRipple(el, e, 'rgba(0,212,170,0.15)'));
  }


  /* ════════════════════════════════════════════════════════════
     INITIALISE ALL ELEMENTS
  ════════════════════════════════════════════════════════════ */
  function initAll() {
    // Primary buttons — magnetic + ripple
    document.querySelectorAll('.btn-primary').forEach(initMagnetic);

    // Ghost buttons — shimmer + ripple
    document.querySelectorAll('.btn-ghost, .btn-danger, .btn-success, .btn-violet').forEach(addShimmer);

    // Stat cards — tilt + shine + particles
    document.querySelectorAll('.stat-card').forEach(initStatCard);

    // Feature cards — tilt + mouse orb + aurora border
    document.querySelectorAll('.feature-card').forEach(el => {
      initTilt(el, 10);
      addFeatureOrb(el);
      initAuroraBorder(el);
    });

    // Card panels — aurora border trace
    document.querySelectorAll('.card, .chart-card').forEach(el => {
      initAuroraBorder(el);
    });

    // Nav items — scan line
    document.querySelectorAll('.nav-item').forEach(el => {
      addScanLine(el);
      const scan = document.createElement('div');
      scan.className = 'nav-scan';
      el.appendChild(scan);
    });

    // Emergency rows — scan + pulse ring
    document.querySelectorAll('.emerg-row').forEach(initEmergRow);

    // Example buttons — pulse ring
    document.querySelectorAll('.example-btn').forEach(addPulseRing);

    // Drug cards — ripple
    document.querySelectorAll('.drug-card').forEach(el => {
      el.addEventListener('mousedown', (e) => addRipple(el, e, 'rgba(0,212,170,0.15)'));
    });

    // DB modal close btn
    document.querySelectorAll('.db-close-btn').forEach(el => {
      el.addEventListener('mousedown', (e) => addRipple(el, e, 'rgba(255,69,96,0.3)'));
    });

    // Model-info items
    document.querySelectorAll('.model-info-item').forEach(el => {
      initAuroraBorder(el);
      el.addEventListener('mousedown', (e) => addRipple(el, e, 'rgba(0,212,170,0.15)'));
    });
  }

  // Run after DOM + Lucide icons ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    // Re-run when lucide finishes (it's deferred)
    setTimeout(initAll, 150);
  }

  // Re-init after any DOM changes (modal opens, etc.)
  const observer = new MutationObserver(() => {
    document.querySelectorAll('.emerg-row:not(.scan-host)').forEach(initEmergRow);
    document.querySelectorAll('.example-btn:not([data-fx])').forEach(el => {
      el.setAttribute('data-fx','1');
      addPulseRing(el);
    });
    document.querySelectorAll('.drug-card:not([data-fx])').forEach(el => {
      el.setAttribute('data-fx','1');
      el.addEventListener('mousedown', (e) => addRipple(el, e, 'rgba(0,212,170,0.15)'));
    });
  });
  observer.observe(document.body, { childList: true, subtree: true });

})();
