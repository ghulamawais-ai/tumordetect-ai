// TumorDetect AI - 3D Brain Background
// Uses Three.js to render a rotating 3D brain-like mesh

(function () {
  const canvas = document.getElementById('brainCanvas');
  if (!canvas || typeof THREE === 'undefined') return;

  // Scene setup
  const scene = new THREE.Scene();

  // Camera
  const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  camera.position.z = 6;

  // Renderer
  const renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    alpha: true,
    antialias: true
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);

  // Create brain-like geometry (icosahedron with noise for organic look)
  const geometry = new THREE.IcosahedronGeometry(2, 4);

  // Distort vertices to look more brain-like (lumpy)
  const positionAttribute = geometry.attributes.position;
  const vertex = new THREE.Vector3();
  for (let i = 0; i < positionAttribute.count; i++) {
    vertex.fromBufferAttribute(positionAttribute, i);
    const noise =
      Math.sin(vertex.x * 3) * 0.1 +
      Math.cos(vertex.y * 3) * 0.1 +
      Math.sin(vertex.z * 3) * 0.1;
    vertex.multiplyScalar(1 + noise);
    positionAttribute.setXYZ(i, vertex.x, vertex.y, vertex.z);
  }
  geometry.computeVertexNormals();

  // Wireframe material (cool line effect)
  const wireGeometry = new THREE.EdgesGeometry(geometry);
  const wireMaterial = new THREE.LineBasicMaterial({
    color: 0x3B82F6,
    transparent: true,
    opacity: 0.6,
    linewidth: 1
  });
  const wireframe = new THREE.LineSegments(wireGeometry, wireMaterial);

  // Glowing points at vertices
  const pointsMaterial = new THREE.PointsMaterial({
    color: 0x8B5CF6,
    size: 0.05,
    transparent: true,
    opacity: 0.8,
    sizeAttenuation: true
  });
  const points = new THREE.Points(geometry, pointsMaterial);

  // Outer soft sphere (aura)
  const auraGeometry = new THREE.SphereGeometry(2.8, 32, 32);
  const auraMaterial = new THREE.MeshBasicMaterial({
    color: 0x06B6D4,
    transparent: true,
    opacity: 0.05,
    side: THREE.BackSide
  });
  const aura = new THREE.Mesh(auraGeometry, auraMaterial);

  // Group everything
  const brainGroup = new THREE.Group();
  brainGroup.add(wireframe);
  brainGroup.add(points);
  brainGroup.add(aura);
  scene.add(brainGroup);

  // Add orbiting particles around brain
  const particleCount = 100;
  const particlesGeometry = new THREE.BufferGeometry();
  const particlePositions = new Float32Array(particleCount * 3);

  for (let i = 0; i < particleCount; i++) {
    const radius = 3.5 + Math.random() * 1.5;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);

    particlePositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
    particlePositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
    particlePositions[i * 3 + 2] = radius * Math.cos(phi);
  }

  particlesGeometry.setAttribute(
    'position',
    new THREE.BufferAttribute(particlePositions, 3)
  );

  const particlesMaterial = new THREE.PointsMaterial({
    color: 0xEC4899,
    size: 0.04,
    transparent: true,
    opacity: 0.6
  });

  const orbitingParticles = new THREE.Points(particlesGeometry, particlesMaterial);
  scene.add(orbitingParticles);

  // Mouse tracking for parallax
  let mouseX = 0;
  let mouseY = 0;
  document.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX / window.innerWidth - 0.5) * 0.3;
    mouseY = (e.clientY / window.innerHeight - 0.5) * 0.3;
  });

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);

    // Rotate brain
    brainGroup.rotation.y += 0.003;
    brainGroup.rotation.x += 0.001;

    // Rotate orbiting particles (opposite direction)
    orbitingParticles.rotation.y -= 0.002;
    orbitingParticles.rotation.x += 0.001;

    // Subtle pulse effect
    const time = Date.now() * 0.001;
    const scale = 1 + Math.sin(time) * 0.02;
    brainGroup.scale.set(scale, scale, scale);

    // Mouse parallax
    brainGroup.rotation.y += (mouseX - brainGroup.rotation.y) * 0.01;
    brainGroup.rotation.x += (-mouseY - brainGroup.rotation.x) * 0.01;

    renderer.render(scene, camera);
  }

  animate();

  // Handle resize
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
})();