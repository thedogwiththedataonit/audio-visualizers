# Converting GLSL Code to Audio-Reactive React Components

This guide documents the complete process of transforming GLSL shader code (like ShaderToy shaders) into fully functional, audio-reactive React components with visual controls.

## Table of Contents
1. [Understanding GLSL/ShaderToy Format](#understanding-glslshadertoy-format)
2. [Component Architecture](#component-architecture)
3. [Audio Analysis System](#audio-analysis-system)
4. [Shader Conversion Process](#shader-conversion-process)
5. [Adding Audio Reactivity](#adding-audio-reactivity)
6. [Designing Visual Controls](#designing-visual-controls)
7. [UI/UX Considerations](#uiux-considerations)
8. [Best Practices](#best-practices)
9. [Common Patterns](#common-patterns)

## Understanding GLSL/ShaderToy Format

### ShaderToy Uniforms
ShaderToy provides these standard uniforms:
```glsl
uniform vec3      iResolution;           // viewport resolution (in pixels)
uniform float     iTime;                 // shader playback time (in seconds)
uniform float     iTimeDelta;            // render time (in seconds)
uniform float     iFrameRate;            // shader frame rate
uniform int       iFrame;                // shader playback frame
uniform float     iChannelTime[4];       // channel playback time (in seconds)
uniform vec3      iChannelResolution[4]; // channel resolution (in pixels)
uniform vec4      iMouse;                // mouse pixel coords
uniform samplerXX iChannel0..3;          // input channel
uniform vec4      iDate;                 // (year, month, day, time in seconds)
uniform float     iSampleRate;           // sound sample rate (i.e., 44100)
```

### Typical ShaderToy Structure
```glsl
// Helper functions (palette, noise, etc.)
vec3 palette(float t) {
    // Color generation logic
}

// Main image function
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Normalize coordinates
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    
    // Shader logic
    vec3 finalColor = vec3(0.0);
    // ... calculations ...
    
    fragColor = vec4(finalColor, 1.0);
}
```

## Component Architecture

### Core Structure
```typescript
const ComponentName: React.FC = () => {
  // Canvas and WebGL refs
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const programRef = useRef<WebGLProgram | null>(null)
  
  // Audio system refs
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioElementRef = useRef<HTMLAudioElement | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null)
  
  // State management
  const [isPlaying, setIsPlaying] = useState(false)
  const [volume, setVolume] = useState(0.7)
  
  // Visual controls
  const [controlName, setControlName] = useState(defaultValue)
  
  // Audio data
  const smoothedAudioDataRef = useRef<AudioData>({
    level: 0,
    bassLevel: 0,
    midLevel: 0,
    trebleLevel: 0,
    frequencyData: new Float32Array(64),
    waveformData: new Float32Array(32)
  })
}
```

## Audio Analysis System

### 1. Audio Data Structure
```typescript
interface AudioData {
  level: number         // Overall volume (0-1)
  bassLevel: number     // Low frequencies (0-1)
  midLevel: number      // Mid frequencies (0-1)
  trebleLevel: number   // High frequencies (0-1)
  frequencyData: Float32Array  // Detailed frequency bins
  waveformData: Float32Array   // Waveform data
}
```

### 2. Frequency Analysis
```typescript
const analyzeAudio = useCallback(() => {
  const analyser = analyserRef.current
  if (!analyser) return smoothedAudioDataRef.current

  const bufferLength = analyser.frequencyBinCount
  const dataArray = new Uint8Array(bufferLength)
  analyser.getByteFrequencyData(dataArray)

  // Frequency band calculation
  const bassEnd = Math.floor(bufferLength * 0.1)    // 0-10%
  const midEnd = Math.floor(bufferLength * 0.5)     // 10-50%
  // Treble: 50-100%

  // Calculate averages for each band
  let bassSum = 0, midSum = 0, trebleSum = 0
  
  for (let i = 0; i < bassEnd; i++) {
    bassSum += dataArray[i]
  }
  bassSum /= bassEnd
  
  // ... similar for mid and treble ...
  
  // Normalize to 0-1 range
  const bassLevel = bassSum / 255
  
  // Apply smoothing
  const smoothingFactor = 0.85
  current.bassLevel = current.bassLevel * smoothingFactor + 
                      bassLevel * (1 - smoothingFactor)
}, [])
```

### 3. Web Audio API Setup
```typescript
const initAudioContext = async () => {
  audioContextRef.current = new AudioContext()
  
  // Create audio graph
  const source = audioContextRef.current.createMediaElementSource(audio)
  const analyser = audioContextRef.current.createAnalyser()
  
  // Configure analyser
  analyser.fftSize = 512  // Frequency resolution
  analyser.smoothingTimeConstant = 0.3  // Internal smoothing
  
  // Connect nodes
  source.connect(analyser)
  analyser.connect(audioContextRef.current.destination)
}
```

## Shader Conversion Process

### 1. Convert to WebGL2 Format

#### Vertex Shader (Standard)
```glsl
#version 300 es
precision highp float;
in vec4 a_position;
void main() {
  gl_Position = a_position;
}
```

#### Fragment Shader Conversion
```glsl
#version 300 es
precision highp float;

out vec4 fragColor;  // Instead of gl_FragColor
uniform vec2 iResolution;  // Keep familiar names
uniform float iTime;

// Add audio uniforms
uniform float u_audioLevel;
uniform float u_bassLevel;
uniform float u_midLevel;
uniform float u_trebleLevel;

// Add control uniforms
uniform float u_controlName;

void main() {
  // Convert from mainImage format
  vec2 fragCoord = gl_FragCoord.xy;
  
  // Original shader logic here
  vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
  
  // ... rest of shader ...
  
  fragColor = vec4(finalColor, 1.0);
}
```

### 2. Uniform Mapping
```typescript
// Standard uniforms
gl.uniform2f(resolutionUniformLocation, canvas.width, canvas.height)
gl.uniform1f(timeUniformLocation, deltaTime)

// Audio uniforms
gl.uniform1f(audioLevelUniformLocation, audioData.level)
gl.uniform1f(bassLevelUniformLocation, audioData.bassLevel)

// Control uniforms
gl.uniform1f(controlUniformLocation, controlValue)
```

## Adding Audio Reactivity

### 1. Identify Visual Parameters
Look for key visual elements that can be modulated:
- **Scale/Zoom**: Good for bass response
- **Rotation Speed**: Mid frequencies work well
- **Color/Hue**: Treble can create sparkle
- **Intensity/Brightness**: Overall level
- **Iteration Count**: Complexity based on energy

### 2. Audio Integration Patterns

#### Pattern 1: Direct Modulation
```glsl
// Scale with bass
float audioZoom = u_zoomLevel + u_bassLevel * 0.5 * u_audioIntensity;
uv = fract(uv * audioZoom) - 0.5;
```

#### Pattern 2: Multiplicative Enhancement
```glsl
// Brightness boost
finalColor *= 1.0 + u_audioLevel * 0.3 * u_audioIntensity;
```

#### Pattern 3: Additive Effects
```glsl
// Color shift
vec3 d = vec3(0.263 + u_colorShift + u_trebleLevel * 0.1, ...);
```

#### Pattern 4: Beat Detection
```glsl
// Pulse on bass hits
float beatPulse = 1.0 + u_bassLevel * u_bassLevel * 0.5 * u_audioIntensity;
finalColor *= beatPulse;
```

### 3. Smooth Transitions
Always include an intensity control:
```glsl
// User can control how much audio affects the visual
float audioEffect = audioValue * u_audioIntensity;
```

## Designing Visual Controls

### 1. Control Selection Strategy

#### For Abstract/Fractal Shaders:
- **Audio Intensity**: Master control for all audio effects
- **Zoom/Scale**: Changes perspective
- **Animation Speed**: Time-based effects
- **Color Shift**: Hue rotation or palette offset
- **Complexity**: Iteration count or detail level

#### For Geometric Shaders:
- **Rotation Speed**: For spinning elements
- **Shape Count**: Number of elements
- **Symmetry**: Geometric patterns
- **Distortion**: Warping effects

#### For Particle/Flow Shaders:
- **Flow Speed**: Particle movement
- **Density**: Number of particles
- **Turbulence**: Chaos level
- **Trail Length**: Motion blur effects

### 2. Control Ranges
```typescript
// Good default ranges
const controls = {
  audioIntensity: { min: 0, max: 2, default: 0.5 },
  zoomLevel: { min: 0.5, max: 3, default: 1.5 },
  animationSpeed: { min: 0, max: 2, default: 1 },
  colorShift: { min: -1, max: 1, default: 0 },
  glowIntensity: { min: 0.5, max: 2, default: 1.2 }
}
```

### 3. Control Implementation
```typescript
// State
const [controlName, setControlName] = useState(defaultValue)

// UI
<input
  type="range"
  min={min * 100}
  max={max * 100}
  value={controlName * 100}
  onChange={(e) => setControlName(parseFloat(e.target.value) / 100)}
/>

// Pass to shader
gl.uniform1f(controlNameUniformLocation, controlName)
```

## UI/UX Considerations

### 1. Layout Principles
- **Compact**: Don't obscure the visualization
- **Organized**: Group related controls
- **Responsive**: Work on all screen sizes
- **Accessible**: Clear labels and good contrast

### 2. Visual Hierarchy
```css
/* Example styling approach */
{
  /* Container */
  background: "rgba(0, 0, 0, 0.85)",
  backdropFilter: "blur(20px)",
  borderRadius: "12px",
  
  /* Sections */
  borderTop: "1px solid rgba(255, 255, 255, 0.15)",
  
  /* Interactive elements */
  transition: "all 0.2s",
  cursor: "pointer"
}
```

### 3. Control Grouping
1. **Audio Controls**: Upload, play/pause, volume, timeline
2. **Visual Controls**: All shader-specific parameters
3. **Presets**: Quick access to interesting combinations
4. **Info Display**: Current audio levels for debugging

## Best Practices

### 1. Performance Optimization
- Use `requestAnimationFrame` for render loop
- Limit frequency analysis to needed resolution
- Implement proper cleanup in `useEffect`
- Reuse typed arrays instead of creating new ones

### 2. Audio Responsiveness
- Balance between reactivity and smoothness
- Use different smoothing factors for different frequencies
- Implement threshold-based effects for beats
- Consider logarithmic scaling for frequency data

### 3. User Experience
- Provide sensible defaults
- Include reset functionality
- Show audio levels for feedback
- Support drag-and-drop for files
- Include sample tracks for testing

### 4. Code Organization
```typescript
// Logical grouping
// 1. Type definitions
// 2. Component setup and refs
// 3. Audio system functions
// 4. File handling
// 5. WebGL setup
// 6. Render function
// 7. UI components
```

## Common Patterns

### 1. Shader Transformation Checklist
- [ ] Convert `mainImage` to `main`
- [ ] Replace `fragColor` output
- [ ] Add `#version 300 es`
- [ ] Define all uniforms
- [ ] Add audio uniforms
- [ ] Add control uniforms
- [ ] Identify modulation points
- [ ] Test with various audio

### 2. Audio Reactivity Checklist
- [ ] Implement frequency analysis
- [ ] Add smoothing
- [ ] Create audio uniforms
- [ ] Add intensity control
- [ ] Test with different music genres
- [ ] Balance visual impact

### 3. Component Structure Template
```typescript
// 1. Imports and interfaces
// 2. Component definition
// 3. Refs and state
// 4. Audio analysis
// 5. File handling
// 6. WebGL effect
// 7. Cleanup effect
// 8. Render JSX
```

### 4. Debugging Tips
- Log audio levels to console
- Use color coding for frequency ranges
- Implement visual audio meters
- Test with sine wave generators
- Check for WebGL errors

## Example Conversion

### Original ShaderToy Code:
```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    float d = length(uv);
    d = sin(d * 8.0 + iTime) / 8.0;
    d = abs(d);
    d = pow(0.01 / d, 1.2);
    vec3 col = vec3(d);
    fragColor = vec4(col, 1.0);
}
```

### Audio-Reactive Version:
```glsl
void main() {
    vec2 uv = (gl_FragCoord.xy * 2.0 - iResolution.xy) / iResolution.y;
    
    // Audio zoom
    uv *= 1.0 + u_bassLevel * 0.5 * u_audioIntensity;
    
    float d = length(uv);
    
    // Audio-modulated frequency
    float freq = 8.0 + u_trebleLevel * 4.0 * u_audioIntensity;
    d = sin(d * freq + iTime * u_animationSpeed) / freq;
    d = abs(d);
    
    // Audio-reactive glow
    float glow = u_glowIntensity + u_midLevel * 0.3 * u_audioIntensity;
    d = pow(0.01 / d, glow);
    
    // Beat pulse
    d *= 1.0 + u_bassLevel * u_bassLevel * 0.5 * u_audioIntensity;
    
    vec3 col = vec3(d);
    fragColor = vec4(col, 1.0);
}
```

This guide provides a complete framework for consistently converting GLSL shaders into engaging, audio-reactive React components with intuitive controls. 