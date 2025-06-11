"use client"

import type React from "react"
import { useRef, useEffect, useState, useCallback } from "react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { X, Settings } from "lucide-react"

interface AudioData {
  level: number
  bassLevel: number
  midLevel: number
  trebleLevel: number
  frequencyData: Float32Array
  waveformData: Float32Array
}

// Update the component to include state management and UI controls
const SnowGLSLVisualization: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [showControls, setShowControls] = useState(false)
  const [colorMode, setColorMode] = useState("default")
  const [visionEffect, setVisionEffect] = useState("normal")
  const [effectIntensity, setEffectIntensity] = useState(0.5)
  const [hueShift, setHueShift] = useState(0.0)
  const [animationSpeed, setAnimationSpeed] = useState(1.0)
  const [zoomLevel, setZoomLevel] = useState(1.0)
  const [rotationSpeed, setRotationSpeed] = useState(1.0)
  const [complexity, setComplexity] = useState(12)
  const [showFps, setShowFps] = useState(false)

  // Audio state
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.7)
  const [trackName, setTrackName] = useState("")
  const [isDragOver, setIsDragOver] = useState(false)

  // Audio control state
  const [audioIntensity, setAudioIntensity] = useState(0.5)
  const [bassInfluence, setBassInfluence] = useState(0.5)
  const [midInfluence, setMidInfluence] = useState(0.5)
  const [trebleInfluence, setTrebleInfluence] = useState(0.5)
  const [beatSensitivity, setBeatSensitivity] = useState(0.5)
  const [smoothing, setSmoothing] = useState(0.85)

  // Audio refs
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioElementRef = useRef<HTMLAudioElement | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null)
  const animationRef = useRef<number>(0)

  // Audio data arrays
  const frequencyDataRef = useRef(new Float32Array(64))
  const waveformDataRef = useRef(new Float32Array(32))
  const smoothedAudioDataRef = useRef<AudioData>({
    level: 0,
    bassLevel: 0,
    midLevel: 0,
    trebleLevel: 0,
    frequencyData: new Float32Array(64),
    waveformData: new Float32Array(32)
  })

  const gl = useRef<WebGL2RenderingContext | null>(null)
  const program = useRef<WebGLProgram | null>(null)
  const positionBuffer = useRef<WebGLBuffer | null>(null)
  const resolutionUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const mouseUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const timeUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const hueShiftUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const colorModeUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const visionEffectUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const effectIntensityUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const fpsRef = useRef<HTMLDivElement>(null)
  const lastTimeRef = useRef<number>(0)
  const frameCountRef = useRef<number>(0)
  const animationSpeedUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const zoomLevelUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const rotationSpeedUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const complexityUniformLocation = useRef<WebGLUniformLocation | null>(null)

  // Audio uniform locations
  const audioLevelUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const bassLevelUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const midLevelUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const trebleLevelUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const frequencyDataUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const waveformDataUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const audioIntensityUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const bassInfluenceUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const midInfluenceUniformLocation = useRef<WebGLUniformLocation | null>(null)
  const trebleInfluenceUniformLocation = useRef<WebGLUniformLocation | null>(null)

  const vertexShaderSource = `#version 300 es
    in vec4 a_position;
    void main() {
      gl_Position = a_position;
    }
  `

  const fragmentShaderSource = `#version 300 es
precision highp float;
out vec4 outColor;
uniform vec2 u_resolution;  // Canvas dimensions in pixels
uniform vec2 u_mouse;       // Mouse position
uniform float u_time;       // Elapsed time in seconds
uniform float u_hueShift;   // Hue shift value
uniform int u_colorMode;    // Color mode (0: default, 1: red, 2: green, 3: purple, 4: rainbow)
uniform int u_visionEffect; // Vision effect (0: normal, 1: invert, 2: grayscale, 3: blur, 4: pixelate)
uniform float u_effectIntensity; // Effect intensity
uniform float u_animationSpeed; // Animation speed multiplier
uniform float u_zoomLevel;      // Zoom level
uniform float u_rotationSpeed;  // Rotation speed multiplier
uniform int u_complexity;       // Fractal complexity

// Audio uniforms
uniform float u_audioLevel;        // Overall audio amplitude (0-1)
uniform float u_bassLevel;         // Bass frequencies (0-1)
uniform float u_midLevel;          // Mid frequencies (0-1)
uniform float u_trebleLevel;       // Treble frequencies (0-1)
uniform float u_frequencyData[64]; // Frequency spectrum array
uniform float u_waveformData[32];  // Time domain waveform
uniform float u_audioIntensity;    // User-controlled audio responsiveness
uniform float u_bassInfluence;     // Bass influence multiplier
uniform float u_midInfluence;      // Mid influence multiplier
uniform float u_trebleInfluence;   // Treble influence multiplier

// Get frequency data for a normalized position (0-1)
float getFrequency(float pos) {
  int index = int(pos * 63.0);
  return u_frequencyData[index];
}

/* 
 * HSV to RGB color conversion with audio reactivity
 * h: hue [0,1], s: saturation [0,1], v: value/brightness [0,1]
 * This is a compact implementation using vector operations for efficient parallel computation on the GPU
 */
vec3 hsv(float h,float s,float v){
  // Apply hue shift based on color mode with audio reactivity
  if (u_colorMode == 1) { // Red/Orange
    h = 0.05 + u_trebleLevel * 0.1 * u_audioIntensity * u_trebleInfluence;
  } else if (u_colorMode == 2) { // Green
    h = 0.3 + u_midLevel * 0.1 * u_audioIntensity * u_midInfluence;
  } else if (u_colorMode == 3) { // Purple
    h = 0.8 + u_bassLevel * 0.1 * u_audioIntensity * u_bassInfluence;
  } else if (u_colorMode == 4) { // Rainbow
    h = fract(h + u_time * 0.1 + u_audioLevel * 0.2 * u_audioIntensity);
  } else {
    // Default: apply hue shift with audio reactivity
    h = fract(h + u_hueShift + u_audioLevel * 0.15 * u_audioIntensity);
  }
  
  // Audio-reactive saturation and brightness
  s = s + u_audioLevel * 0.3 * u_audioIntensity;
  v = v * (1.0 + u_bassLevel * u_bassLevel * 0.5 * u_audioIntensity * u_bassInfluence);
  
  vec4 t=vec4(1.,2./3.,1./3.,3.);
  vec3 p=abs(fract(vec3(h)+t.xyz)*6.-vec3(t.w));
  return v*mix(vec3(t.x),clamp(p-vec3(t.x),0.,1.),s);
}

/*
 * 2D rotation matrix around origin
 * Used for rotating points in the XZ plane
 * Equivalent to the signal processing operation of phase shifting
 */
mat2 rotate2D(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat2(c, -s, s, c);
}

/*
 * 3D rotation matrix around arbitrary axis
 * Implements Rodrigues' rotation formula
 * This is similar to a 3D filter kernel with directional preference
 */
mat3 rotate3D(float angle, vec3 axis) {
  axis = normalize(axis);
  float s = sin(angle);
  float c = cos(angle);
  float oc = 1.0 - c;
  
  return mat3(
    oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
    oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
    oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c
  );
}

// Apply pixelate effect
vec2 pixelate(vec2 uv, float pixelSize) {
  return floor(uv / pixelSize) * pixelSize;
}

// Apply blur effect (simple box blur)
vec4 blur(sampler2D tex, vec2 uv, vec2 resolution, float intensity) {
  float blurSize = intensity * 0.01;
  vec4 color = vec4(0.0);
  for(float i = -2.0; i <= 2.0; i++) {
    for(float j = -2.0; j <= 2.0; j++) {
      color += texture(tex, uv + vec2(i, j) * blurSize / resolution);
    }
  }
  return color / 25.0;
}

void main() {
  // Initialize variables
  vec2 r = u_resolution;                // Screen resolution
  vec2 FC = gl_FragCoord.xy;            // Current pixel coordinates
  float t = u_time;                     // Animation time
  vec4 o = vec4(0,0,0,1);               // Output color (initially black with alpha=1)
  
  // Apply pixelate effect to coordinates if needed
  if (u_visionEffect == 4) { // Pixelate
    float pixelSize = mix(1.0, 20.0, u_effectIntensity);
    FC = pixelate(FC, pixelSize);
  }
  
  // Audio-reactive loop count
  float audioPulse = 1.0 + u_audioLevel * 0.5 * u_audioIntensity;
  float maxIterations = 99.0 * audioPulse;
  
  /* 
   * Main rendering loop - accumulates color contributions with audio reactivity
   * This implements a form of path tracing with variable samples per pixel based on audio
   * i: loop counter, g: accumulated distance, e: escape multiplier, s: scale factor
   */
  for(float i=0.,g=0.,e=0.,s=0.;++i<maxIterations;){
    /* 
     * Transform pixel coordinates to 3D space with audio reactivity
     * 1. Normalize coordinates to [-2,2] range based on screen aspect ratio
     * 2. Apply 3D rotation around axis (0,9,-3) with audio-reactive angle
     * 3. Apply time-varying rotation in XZ plane with audio-reactive speed
     * 
     * In signal processing terms, this is applying a spatiotemporal 
     * transformation to the input coordinates (similar to a warping filter)
     */
    float audioRotation = 3.0 + u_bassLevel * 2.0 * u_audioIntensity * u_bassInfluence;
    vec3 p=vec3((FC.xy-.5*r)/r.y*4.+vec2(0,1),g-6.)*rotate3D(audioRotation,vec3(0,9,-3));
    
    // Audio-reactive zoom and positioning
    float audioZoom = u_zoomLevel * (1.0 + u_audioLevel * 0.3 * u_audioIntensity);
    p = p / audioZoom;
    
    // Audio-reactive rotation with different frequencies affecting different axes
    float bassRotSpeed = t * 0.3 * u_rotationSpeed * (1.0 + u_bassLevel * u_bassInfluence * u_audioIntensity);
    float midRotSpeed = t * 0.2 * u_rotationSpeed * u_midLevel * u_midInfluence * u_audioIntensity;
    p.xz *= rotate2D(bassRotSpeed);
    p.yz *= rotate2D(midRotSpeed);
    
    /* 
     * Initial scale factor for the fractal with audio reactivity
     * This controls the overall "zoom level" of the fractal
     */
    s = 6.0 * (1.0 + u_audioLevel * 0.3 * u_audioIntensity);
    
    // Audio-reactive complexity
    int audioComplexity = u_complexity + int(u_audioLevel * 8.0 * u_audioIntensity);
    
    /* 
     * Audio-reactive fractal iteration loop (Mandelbox/Mandelbulb variant)
     * This is a distance estimation algorithm that:
     * 1. Folds space through absolute value operations (creating symmetry)
     * 2. Scales by a factor e (creating self-similarity)
     * 3. Translates by audio-reactive offset (offsetting the pattern)
     * 
     * The dot product operation (dot(p,p*.47)) creates a radial distance field
     * which produces noise patterns with the following characteristics:
     * - Frequency: Controlled by the .47 scaling factor (higher = more detail)
     * - Distribution: Approximately Gaussian due to the dot product
     * - Self-similarity: Due to the recursive application in the loop
     * 
     * In signal processing terms, this is a non-linear recursive filter
     * that generates coherent noise with fractal characteristics
     */
    vec3 audioOffset = vec3(0.0, 4.03 + u_midLevel * 1.0 * u_audioIntensity * u_midInfluence, -1.0 + u_trebleLevel * 0.5 * u_audioIntensity * u_trebleInfluence);
    vec3 audioFold = vec3(3.0 + u_bassLevel * 0.5 * u_audioIntensity * u_bassInfluence, 4.0, 3.0);
    float audioScale = 7.5 + u_audioLevel * 2.0 * u_audioIntensity;
    
    for(int i=0;i++<audioComplexity;p=audioOffset-abs(abs(p)*e-audioFold))
      s*=e=audioScale/dot(p,p*.47);
    
    /* 
     * Accumulate distance field with audio reactivity
     * g += p.y*p.y/s*audioFactor: Adds squared y-component scaled by inverse of s and audio
     * This creates a depth-dependent accumulation with audio modulation
     * 
     * In image processing terms, this is similar to a depth-weighted 
     * accumulation buffer or a form of volumetric integration
     */
    float audioDepthFactor = 0.3 * (1.0 + u_audioLevel * 0.2 * u_audioIntensity);
    g += p.y * p.y / s * audioDepthFactor;
    
    /* 
     * Calculate color intensity with audio reactivity
     * log2(s)-g*audioGFactor: Logarithmic scale factor minus accumulated distance
     * This creates a balance between detail (log2(s)) and depth (g) with audio modulation
     * 
     * In signal processing, this is similar to applying a logarithmic 
     * response curve (like in audio processing) with a linear offset
     */
    float audioGFactor = 0.8 * (1.0 + u_midLevel * 0.3 * u_audioIntensity * u_midInfluence);
    s = log2(s) - g * audioGFactor;
    
    /* 
     * Add color contribution using HSV color space with audio reactivity
     * - Hue: Audio-reactive based on frequency spectrum
     * - Saturation: Audio-reactive for more vibrant colors during peaks
     * - Value: Based on s scaled with audio intensity
     * 
     * The resulting noise pattern has characteristics of:
     * - Audio-reactive color variation
     * - Fractal detail (self-similarity at different scales)
     * - Volumetric appearance with audio modulation
     */
    float freqPos = i / maxIterations;
    float freqIntensity = getFrequency(freqPos);
    float audioHue = 0.5 + freqIntensity * 0.2 * u_audioIntensity;
    float audioSat = 0.1 + u_audioLevel * 0.4 * u_audioIntensity;
    float audioVal = s / (7e2 - u_audioLevel * 200.0 * u_audioIntensity);
    
    // Beat-reactive brightness pulse
    float beatPulse = 1.0 + u_bassLevel * u_bassLevel * 0.3 * u_audioIntensity * u_bassInfluence;
    audioVal *= beatPulse;
    
    o.rgb += hsv(audioHue, audioSat, audioVal);
  }
  
  // Apply vision effects
  if (u_visionEffect == 1) { // Invert
    o.rgb = mix(o.rgb, 1.0 - o.rgb, u_effectIntensity);
  } else if (u_visionEffect == 2) { // Grayscale
    float gray = dot(o.rgb, vec3(0.299, 0.587, 0.114));
    o.rgb = mix(o.rgb, vec3(gray), u_effectIntensity);
  }
  
  // Final output color
  outColor = o;
}
`

  const createShader = (gl: WebGL2RenderingContext, type: number, source: string) => {
    const shader = gl.createShader(type)
    if (!shader) {
      console.error("Failed to create shader")
      return null
    }
    gl.shaderSource(shader, source)
    gl.compileShader(shader)

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error("Shader compilation error:", gl.getShaderInfoLog(shader))
      gl.deleteShader(shader)
      return null
    }

    return shader
  }

  // Initialize audio context
  const initAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()

      // Resume context if suspended (browser autoplay policy)
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume()
      }
    }
  }, [])

  // Audio analysis function
  const analyzeAudio = useCallback(() => {
    const analyser = analyserRef.current
    if (!analyser) return smoothedAudioDataRef.current

    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)
    const waveformArray = new Uint8Array(bufferLength)

    analyser.getByteFrequencyData(dataArray)
    analyser.getByteTimeDomainData(waveformArray)

    // Calculate frequency bands
    const bassEnd = Math.floor(bufferLength * 0.1)
    const midEnd = Math.floor(bufferLength * 0.5)

    let bassSum = 0, midSum = 0, trebleSum = 0, totalSum = 0

    // Bass (0-10% of frequency range)
    for (let i = 0; i < bassEnd; i++) {
      bassSum += dataArray[i]
    }
    bassSum /= bassEnd

    // Mid (10-50% of frequency range)
    for (let i = bassEnd; i < midEnd; i++) {
      midSum += dataArray[i]
    }
    midSum /= (midEnd - bassEnd)

    // Treble (50-100% of frequency range)
    for (let i = midEnd; i < bufferLength; i++) {
      trebleSum += dataArray[i]
    }
    trebleSum /= (bufferLength - midEnd)

    // Overall level
    for (let i = 0; i < bufferLength; i++) {
      totalSum += dataArray[i]
    }
    const level = totalSum / bufferLength / 255

    // Normalize frequency bands
    const bassLevel = bassSum / 255
    const midLevel = midSum / 255
    const trebleLevel = trebleSum / 255

    // Downsample frequency data to 64 bins
    for (let i = 0; i < 64; i++) {
      const index = Math.floor((i / 64) * bufferLength)
      frequencyDataRef.current[i] = dataArray[index] / 255
    }

    // Downsample waveform data to 32 bins
    for (let i = 0; i < 32; i++) {
      const index = Math.floor((i / 32) * bufferLength)
      waveformDataRef.current[i] = (waveformArray[index] - 128) / 128
    }

    // Smooth the audio data to prevent jarring transitions
    const smoothingFactor = smoothing
    const current = smoothedAudioDataRef.current

    current.level = current.level * smoothingFactor + level * (1 - smoothingFactor)
    current.bassLevel = current.bassLevel * smoothingFactor + bassLevel * (1 - smoothingFactor)
    current.midLevel = current.midLevel * smoothingFactor + midLevel * (1 - smoothingFactor)
    current.trebleLevel = current.trebleLevel * smoothingFactor + trebleLevel * (1 - smoothingFactor)

    // Update frequency and waveform arrays
    for (let i = 0; i < 64; i++) {
      current.frequencyData[i] = current.frequencyData[i] * smoothingFactor + frequencyDataRef.current[i] * (1 - smoothingFactor)
    }

    for (let i = 0; i < 32; i++) {
      current.waveformData[i] = current.waveformData[i] * smoothingFactor + waveformDataRef.current[i] * (1 - smoothingFactor)
    }

    return current
  }, [smoothing])

  // Load audio file from File object
  const loadAudioFile = useCallback(async (file: File) => {
    await initAudioContext()

    const audio = new Audio()
    const url = URL.createObjectURL(file)
    audio.src = url
    audio.crossOrigin = "anonymous"
    audio.volume = volume

    // Set up audio element event listeners
    audio.addEventListener('loadedmetadata', () => {
      setDuration(audio.duration)
      setTrackName(file.name.replace(/\.[^/.]+$/, ""))
    })

    audio.addEventListener('timeupdate', () => {
      setCurrentTime(audio.currentTime)
    })

    audio.addEventListener('ended', () => {
      setIsPlaying(false)
    })

    // Clean up previous audio setup
    if (audioElementRef.current) {
      audioElementRef.current.pause()
      if (audioElementRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioElementRef.current.src)
      }
    }

    audioElementRef.current = audio

    // Set up Web Audio API
    if (audioContextRef.current) {
      if (sourceRef.current) {
        sourceRef.current.disconnect()
      }

      const source = audioContextRef.current.createMediaElementSource(audio)
      const analyser = audioContextRef.current.createAnalyser()

      analyser.fftSize = 512
      analyser.smoothingTimeConstant = 0.3

      source.connect(analyser)
      analyser.connect(audioContextRef.current.destination)

      sourceRef.current = source
      analyserRef.current = analyser
    }
  }, [volume, initAudioContext])

  // Load audio file from URL (for default songs)
  const loadAudioFromUrl = useCallback(async (url: string, filename: string) => {
    await initAudioContext()

    const audio = new Audio()
    audio.src = url
    audio.crossOrigin = "anonymous"
    audio.volume = volume

    // Set up audio element event listeners
    audio.addEventListener('loadedmetadata', () => {
      setDuration(audio.duration)
      setTrackName(filename.replace(/\.[^/.]+$/, ""))
    })

    audio.addEventListener('timeupdate', () => {
      setCurrentTime(audio.currentTime)
    })

    audio.addEventListener('ended', () => {
      setIsPlaying(false)
    })

    // Clean up previous audio setup
    if (audioElementRef.current) {
      audioElementRef.current.pause()
      if (audioElementRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioElementRef.current.src)
      }
    }

    audioElementRef.current = audio

    // Set up Web Audio API
    if (audioContextRef.current) {
      if (sourceRef.current) {
        sourceRef.current.disconnect()
      }

      const source = audioContextRef.current.createMediaElementSource(audio)
      const analyser = audioContextRef.current.createAnalyser()

      analyser.fftSize = 512
      analyser.smoothingTimeConstant = 0.3

      source.connect(analyser)
      analyser.connect(audioContextRef.current.destination)

      sourceRef.current = source
      analyserRef.current = analyser
    }
  }, [volume, initAudioContext])

  // File upload handlers
  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith('audio/')) {
      loadAudioFile(file)
    }
  }, [loadAudioFile])

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    setIsDragOver(false)

    const file = event.dataTransfer.files[0]
    if (file && file.type.startsWith('audio/')) {
      loadAudioFile(file)
    }
  }, [loadAudioFile])

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false)
  }, [])

  // Audio control functions
  const togglePlayPause = useCallback(async () => {
    if (!audioElementRef.current) return

    await initAudioContext()

    if (isPlaying) {
      audioElementRef.current.pause()
      setIsPlaying(false)
    } else {
      try {
        await audioElementRef.current.play()
        setIsPlaying(true)
      } catch (error) {
        console.error('Error playing audio:', error)
      }
    }
  }, [isPlaying, initAudioContext])

  const handleSeek = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioElementRef.current) return

    const seekTime = (parseFloat(event.target.value) / 100) * duration
    audioElementRef.current.currentTime = seekTime
    setCurrentTime(seekTime)
  }, [duration])

  const handleVolumeChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(event.target.value) / 100
    setVolume(newVolume)

    if (audioElementRef.current) {
      audioElementRef.current.volume = newVolume
    }
  }, [])

  const formatTime = useCallback((time: number) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }, [])

  // Default songs available in the app
  const defaultSongs = [
    { filename: "Charli XCX - party 4 u.mp3", displayName: "Charli XCX - Party 4 U" },
    { filename: "Benson Boone - Beautiful Things.mp3", displayName: "Benson Boone - Beautiful Things" },
    { filename: "M83 - Midnight City.mp3", displayName: "M83 - Midnight City" }
  ]

  // Load default song
  const loadDefaultSong = useCallback((filename: string, displayName: string) => {
    const url = `/songs/${encodeURIComponent(filename)}`
    loadAudioFromUrl(url, displayName)
  }, [loadAudioFromUrl])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // Get WebGL2 context
    const context = canvas.getContext("webgl2")
    if (!context) {
      console.error("WebGL2 not supported")
      return
    }
    gl.current = context

    // Resize canvas to full screen
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    // Create and compile shaders
    const vertexShader = createShader(gl.current, gl.current.VERTEX_SHADER, vertexShaderSource)
    const fragmentShader = createShader(gl.current, gl.current.FRAGMENT_SHADER, fragmentShaderSource)

    if (!vertexShader || !fragmentShader) return

    // Create program and link shaders
    const programToUse = gl.current.createProgram()
    if (!programToUse) {
      console.error("Failed to create program")
      return
    }

    gl.current.attachShader(programToUse, vertexShader)
    gl.current.attachShader(programToUse, fragmentShader)
    gl.current.linkProgram(programToUse)

    if (!gl.current.getProgramParameter(programToUse, gl.current.LINK_STATUS)) {
      console.error("Program linking error:", gl.current.getProgramInfoLog(programToUse))
      return
    }

    program.current = programToUse

    // Set up position buffer (full screen quad)
    const positionBufferToUse = gl.current.createBuffer()
    gl.current.bindBuffer(gl.current.ARRAY_BUFFER, positionBufferToUse)
    gl.current.bufferData(
      gl.current.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.current.STATIC_DRAW,
    )

    positionBuffer.current = positionBufferToUse

    // Set up attributes
    const positionAttributeLocation = gl.current.getAttribLocation(programToUse, "a_position")
    gl.current.enableVertexAttribArray(positionAttributeLocation)
    gl.current.vertexAttribPointer(positionAttributeLocation, 2, gl.current.FLOAT, false, 0, 0)

    // Set up uniforms
    const resolutionUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_resolution")
    const mouseUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_mouse")
    const timeUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_time")
    const hueShiftUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_hueShift")
    const colorModeUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_colorMode")
    const visionEffectUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_visionEffect")
    const effectIntensityUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_effectIntensity")
    const animationSpeedUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_animationSpeed")
    const zoomLevelUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_zoomLevel")
    const rotationSpeedUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_rotationSpeed")
    const complexityUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_complexity")
    
    // Audio uniform locations
    const audioLevelUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_audioLevel")
    const bassLevelUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_bassLevel")
    const midLevelUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_midLevel")
    const trebleLevelUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_trebleLevel")
    const frequencyDataUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_frequencyData")
    const waveformDataUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_waveformData")
    const audioIntensityUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_audioIntensity")
    const bassInfluenceUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_bassInfluence")
    const midInfluenceUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_midInfluence")
    const trebleInfluenceUniformLocationToUse = gl.current.getUniformLocation(programToUse, "u_trebleInfluence")

    resolutionUniformLocation.current = resolutionUniformLocationToUse
    mouseUniformLocation.current = mouseUniformLocationToUse
    timeUniformLocation.current = timeUniformLocationToUse
    hueShiftUniformLocation.current = hueShiftUniformLocationToUse
    colorModeUniformLocation.current = colorModeUniformLocationToUse
    visionEffectUniformLocation.current = visionEffectUniformLocationToUse
    effectIntensityUniformLocation.current = effectIntensityUniformLocationToUse
    animationSpeedUniformLocation.current = animationSpeedUniformLocationToUse
    zoomLevelUniformLocation.current = zoomLevelUniformLocationToUse
    rotationSpeedUniformLocation.current = rotationSpeedUniformLocationToUse
    complexityUniformLocation.current = complexityUniformLocationToUse
    
    // Audio uniform locations
    audioLevelUniformLocation.current = audioLevelUniformLocationToUse
    bassLevelUniformLocation.current = bassLevelUniformLocationToUse
    midLevelUniformLocation.current = midLevelUniformLocationToUse
    trebleLevelUniformLocation.current = trebleLevelUniformLocationToUse
    frequencyDataUniformLocation.current = frequencyDataUniformLocationToUse
    waveformDataUniformLocation.current = waveformDataUniformLocationToUse
    audioIntensityUniformLocation.current = audioIntensityUniformLocationToUse
    bassInfluenceUniformLocation.current = bassInfluenceUniformLocationToUse
    midInfluenceUniformLocation.current = midInfluenceUniformLocationToUse
    trebleInfluenceUniformLocation.current = trebleInfluenceUniformLocationToUse

    // Mouse tracking
    let mouseX = 0
    let mouseY = 0
    const handleMouseMove = (e: MouseEvent) => {
      mouseX = e.clientX
      mouseY = e.clientY
    }
    window.addEventListener("mousemove", handleMouseMove)

    // Render loop
    const startTime = performance.now()

    const render = () => {
      const currentTime = performance.now()
      const elapsedTime = (currentTime - startTime) / 1000 // Convert to seconds

      // Update canvas size if needed
      if (canvas.width !== window.innerWidth || canvas.height !== window.innerHeight) {
        resizeCanvas()
      }

      // Check if WebGL context is available
      if (!gl.current) return

      // Set viewport and clear
      gl.current.viewport(0, 0, canvas.width, canvas.height)
      gl.current.clearColor(0, 0, 0, 1)
      gl.current.clear(gl.current.COLOR_BUFFER_BIT)

      // Use our program
      if (program.current) {
        gl.current.useProgram(program.current)
      }

      // Get current audio data
      const audioData = analyzeAudio()

      // Update uniforms
      if (timeUniformLocation.current && gl.current) {
        gl.current.uniform1f(timeUniformLocation.current, elapsedTime * animationSpeed)
      }
      if (animationSpeedUniformLocation.current && gl.current) {
        gl.current.uniform1f(animationSpeedUniformLocation.current, animationSpeed)
      }
      if (zoomLevelUniformLocation.current && gl.current) {
        gl.current.uniform1f(zoomLevelUniformLocation.current, zoomLevel)
      }
      if (rotationSpeedUniformLocation.current && gl.current) {
        gl.current.uniform1f(rotationSpeedUniformLocation.current, rotationSpeed)
      }
      if (complexityUniformLocation.current && gl.current) {
        gl.current.uniform1i(complexityUniformLocation.current, complexity)
      }
      if (resolutionUniformLocation.current && gl.current) {
        gl.current.uniform2f(resolutionUniformLocation.current, canvas.width, canvas.height)
      }
      if (mouseUniformLocation.current && gl.current) {
        gl.current.uniform2f(mouseUniformLocation.current, mouseX, canvas.height - mouseY) // Flip Y for WebGL coords
      }
      if (hueShiftUniformLocation.current && gl.current) {
        gl.current.uniform1f(hueShiftUniformLocation.current, hueShift)
      }

      // Update audio uniforms
      if (audioLevelUniformLocation.current && gl.current) {
        gl.current.uniform1f(audioLevelUniformLocation.current, audioData.level)
      }
      if (bassLevelUniformLocation.current && gl.current) {
        gl.current.uniform1f(bassLevelUniformLocation.current, audioData.bassLevel)
      }
      if (midLevelUniformLocation.current && gl.current) {
        gl.current.uniform1f(midLevelUniformLocation.current, audioData.midLevel)
      }
      if (trebleLevelUniformLocation.current && gl.current) {
        gl.current.uniform1f(trebleLevelUniformLocation.current, audioData.trebleLevel)
      }
      if (frequencyDataUniformLocation.current && gl.current) {
        gl.current.uniform1fv(frequencyDataUniformLocation.current, audioData.frequencyData)
      }
      if (waveformDataUniformLocation.current && gl.current) {
        gl.current.uniform1fv(waveformDataUniformLocation.current, audioData.waveformData)
      }
      if (audioIntensityUniformLocation.current && gl.current) {
        gl.current.uniform1f(audioIntensityUniformLocation.current, audioIntensity)
      }
      if (bassInfluenceUniformLocation.current && gl.current) {
        gl.current.uniform1f(bassInfluenceUniformLocation.current, bassInfluence)
      }
      if (midInfluenceUniformLocation.current && gl.current) {
        gl.current.uniform1f(midInfluenceUniformLocation.current, midInfluence)
      }
      if (trebleInfluenceUniformLocation.current && gl.current) {
        gl.current.uniform1f(trebleInfluenceUniformLocation.current, trebleInfluence)
      }

      // Set color mode
      let colorModeValue = 0
      if (colorMode === "red") colorModeValue = 1
      else if (colorMode === "green") colorModeValue = 2
      else if (colorMode === "purple") colorModeValue = 3
      else if (colorMode === "rainbow") colorModeValue = 4
      if (colorModeUniformLocation.current && gl.current) {
        gl.current.uniform1i(colorModeUniformLocation.current, colorModeValue)
      }

      // Set vision effect
      let visionEffectValue = 0
      if (visionEffect === "invert") visionEffectValue = 1
      else if (visionEffect === "grayscale") visionEffectValue = 2
      else if (visionEffect === "blur") visionEffectValue = 3
      else if (visionEffect === "pixelate") visionEffectValue = 4
      if (visionEffectUniformLocation.current && gl.current) {
        gl.current.uniform1i(visionEffectUniformLocation.current, visionEffectValue)
      }

      // Set effect intensity
      if (effectIntensityUniformLocation.current && gl.current) {
        gl.current.uniform1f(effectIntensityUniformLocation.current, effectIntensity)
      }

      // Draw
      if (gl.current) {
        gl.current.drawArrays(gl.current.TRIANGLES, 0, 6)
      }

      // Calculate FPS
      if (showFps && fpsRef.current) {
        frameCountRef.current++
        const now = performance.now()
        const elapsed = now - lastTimeRef.current
        if (elapsed >= 1000) {
          const fps = Math.round((frameCountRef.current * 1000) / elapsed)
          fpsRef.current.textContent = `${fps} FPS`
          frameCountRef.current = 0
          lastTimeRef.current = now
        }
      }

      // Request next frame
      requestAnimationFrame(render)
    }

    // Start rendering
    render()

    // Cleanup
    return () => {
      window.removeEventListener("resize", resizeCanvas)
      window.removeEventListener("mousemove", handleMouseMove)
      if (program.current && gl.current) gl.current.deleteProgram(program.current)
      if (positionBuffer.current && gl.current) gl.current.deleteBuffer(positionBuffer.current)
    }
  }, [
    colorMode,
    visionEffect,
    effectIntensity,
    hueShift,
    animationSpeed,
    zoomLevel,
    rotationSpeed,
    complexity,
    showFps,
    analyzeAudio,
    audioIntensity,
    bassInfluence,
    midInfluence,
    trebleInfluence,
  ])

  // Cleanup audio on unmount
  useEffect(() => {
    return () => {
      if (audioElementRef.current) {
        audioElementRef.current.pause()
        if (audioElementRef.current.src.startsWith('blob:')) {
          URL.revokeObjectURL(audioElementRef.current.src)
        }
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
      }
    }
  }, [])

  return (
    <div 
      className="relative w-full h-screen"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <canvas ref={canvasRef} className="w-full h-screen block" style={{ display: "block" }} />
      {showFps && (
        <div ref={fpsRef} className="absolute top-4 left-4 bg-black/50 text-white px-2 py-1 rounded font-mono text-sm">
          0 FPS
        </div>
      )}

      {/* Audio Player */}
      <div className="absolute top-20 left-4 bg-black/70 backdrop-blur-md text-white p-4 rounded-lg w-80 shadow-lg">
        <h3 className="text-lg font-semibold mb-4">Audio Player</h3>

        {/* File Upload */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Upload Audio File</label>
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            className="w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-white/20 file:text-white hover:file:bg-white/30 cursor-pointer"
          />
        </div>

        {/* Default Songs */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Or Choose a Default Song</label>
          <div className="space-y-2">
            {defaultSongs.map((song, index) => (
              <button
                key={index}
                onClick={() => loadDefaultSong(song.filename, song.displayName)}
                className="w-full text-left p-2 bg-white/10 hover:bg-white/20 rounded transition-colors text-sm"
              >
                {song.displayName}
              </button>
            ))}
          </div>
        </div>

        {/* Track Info */}
        {trackName && (
          <div className="mb-4 text-sm font-medium text-blue-300">
            {trackName}
          </div>
        )}

        {/* Play/Pause Button */}
        <div className="mb-4 text-center">
          <button
            onClick={togglePlayPause}
            disabled={!audioElementRef.current}
            className={`px-6 py-2 rounded-full font-semibold transition-colors ${
              isPlaying 
                ? "bg-red-500 hover:bg-red-600" 
                : "bg-green-500 hover:bg-green-600"
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {isPlaying ? "⏸ Pause" : "▶ Play"}
          </button>
        </div>

        {/* Timeline */}
        {duration > 0 && (
          <div className="mb-4">
            <div className="flex justify-between text-xs mb-1">
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={(currentTime / duration) * 100}
              onChange={handleSeek}
              className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
            />
          </div>
        )}

        {/* Volume Control */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">
            Volume: {Math.round(volume * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={volume * 100}
            onChange={handleVolumeChange}
            className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
          />
        </div>

        {/* Audio Levels Display */}
        {isPlaying && (
          <div className="text-xs space-y-1">
            <div>Level: {Math.round(smoothedAudioDataRef.current.level * 100)}%</div>
            <div>Bass: {Math.round(smoothedAudioDataRef.current.bassLevel * 100)}%</div>
            <div>Mid: {Math.round(smoothedAudioDataRef.current.midLevel * 100)}%</div>
            <div>Treble: {Math.round(smoothedAudioDataRef.current.trebleLevel * 100)}%</div>
          </div>
        )}
      </div>

      {/* Settings toggle button */}
      <button
        onClick={() => setShowControls(!showControls)}
        className="absolute top-4 right-4 bg-black/50 text-white p-2 rounded-full hover:bg-black/70 transition-colors"
        aria-label={showControls ? "Hide controls" : "Show controls"}
      >
        {showControls ? <X size={20} /> : <Settings size={20} />}
      </button>

      {/* Controls panel */}
      {showControls && (
        <div className="absolute top-16 right-4 bg-black/70 backdrop-blur-md text-white p-4 rounded-lg w-72 shadow-lg max-h-[80vh] overflow-y-auto">
          <h3 className="text-lg font-semibold mb-4">Visual Settings</h3>

          <div className="space-y-4">
            <div className="space-y-2">
              <label className="block text-sm font-medium">Color Mode</label>
              <Select value={colorMode} onValueChange={setColorMode}>
                <SelectTrigger className="bg-black/50 border-gray-700">
                  <SelectValue placeholder="Select color mode" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700 text-white">
                  <SelectItem value="default">Default (Blue)</SelectItem>
                  <SelectItem value="red">Red/Orange</SelectItem>
                  <SelectItem value="green">Green</SelectItem>
                  <SelectItem value="purple">Purple</SelectItem>
                  <SelectItem value="rainbow">Rainbow</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {colorMode === "default" && (
              <div className="space-y-2">
                <label className="block text-sm font-medium">Hue Shift</label>
                <Slider
                  value={[hueShift]}
                  min={0}
                  max={1}
                  step={0.01}
                  onValueChange={(values) => setHueShift(values[0])}
                  className="py-2"
                />
              </div>
            )}

            <div className="space-y-2">
              <label className="block text-sm font-medium">Vision Effect</label>
              <Select value={visionEffect} onValueChange={setVisionEffect}>
                <SelectTrigger className="bg-black/50 border-gray-700">
                  <SelectValue placeholder="Select vision effect" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700 text-white">
                  <SelectItem value="normal">Normal</SelectItem>
                  <SelectItem value="invert">Invert</SelectItem>
                  <SelectItem value="grayscale">Grayscale</SelectItem>
                  <SelectItem value="pixelate">Pixelate</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {visionEffect !== "normal" && (
              <div className="space-y-2">
                <label className="block text-sm font-medium">Effect Intensity</label>
                <Slider
                  value={[effectIntensity]}
                  min={0}
                  max={1}
                  step={0.01}
                  onValueChange={(values) => setEffectIntensity(values[0])}
                  className="py-2"
                />
              </div>
            )}

            <div className="space-y-2">
              <label className="block text-sm font-medium">Animation Speed: {animationSpeed.toFixed(1)}x</label>
              <Slider
                value={[animationSpeed]}
                min={0.1}
                max={3}
                step={0.1}
                onValueChange={(values) => setAnimationSpeed(values[0])}
                className="py-2"
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium">Zoom Level: {zoomLevel.toFixed(1)}x</label>
              <Slider
                value={[zoomLevel]}
                min={0.5}
                max={3}
                step={0.1}
                onValueChange={(values) => setZoomLevel(values[0])}
                className="py-2"
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium">Rotation Speed: {rotationSpeed.toFixed(1)}x</label>
              <Slider
                value={[rotationSpeed]}
                min={0}
                max={3}
                step={0.1}
                onValueChange={(values) => setRotationSpeed(values[0])}
                className="py-2"
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium">Complexity: {complexity}</label>
              <Slider
                value={[complexity]}
                min={6}
                max={24}
                step={1}
                onValueChange={(values) => setComplexity(values[0])}
                className="py-2"
              />
            </div>

            {/* Audio Controls Section */}
            <div className="border-t border-gray-700 pt-4">
              <h4 className="text-md font-semibold mb-3">Audio Reactivity</h4>
              
              <div className="space-y-3">
                <div className="space-y-2">
                  <label className="block text-sm font-medium">Audio Intensity: {Math.round(audioIntensity * 100)}%</label>
                  <Slider
                    value={[audioIntensity]}
                    min={0}
                    max={2}
                    step={0.01}
                    onValueChange={(values) => setAudioIntensity(values[0])}
                    className="py-2"
                  />
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-medium">Bass Influence: {Math.round(bassInfluence * 100)}%</label>
                  <Slider
                    value={[bassInfluence]}
                    min={0}
                    max={2}
                    step={0.01}
                    onValueChange={(values) => setBassInfluence(values[0])}
                    className="py-2"
                  />
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-medium">Mid Influence: {Math.round(midInfluence * 100)}%</label>
                  <Slider
                    value={[midInfluence]}
                    min={0}
                    max={2}
                    step={0.01}
                    onValueChange={(values) => setMidInfluence(values[0])}
                    className="py-2"
                  />
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-medium">Treble Influence: {Math.round(trebleInfluence * 100)}%</label>
                  <Slider
                    value={[trebleInfluence]}
                    min={0}
                    max={2}
                    step={0.01}
                    onValueChange={(values) => setTrebleInfluence(values[0])}
                    className="py-2"
                  />
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-medium">Beat Sensitivity: {Math.round(beatSensitivity * 100)}%</label>
                  <Slider
                    value={[beatSensitivity]}
                    min={0}
                    max={1}
                    step={0.01}
                    onValueChange={(values) => setBeatSensitivity(values[0])}
                    className="py-2"
                  />
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-medium">Audio Smoothing: {Math.round(smoothing * 100)}%</label>
                  <Slider
                    value={[smoothing]}
                    min={0.1}
                    max={0.95}
                    step={0.01}
                    onValueChange={(values) => setSmoothing(values[0])}
                    className="py-2"
                  />
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Show FPS</label>
              <button
                onClick={() => setShowFps(!showFps)}
                className={`px-3 py-1 rounded ${showFps ? "bg-white/20" : "bg-black/40"}`}
              >
                {showFps ? "On" : "Off"}
              </button>
            </div>

            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Fullscreen</label>
              <button
                onClick={() => {
                  if (document.fullscreenElement) {
                    document.exitFullscreen()
                  } else {
                    document.documentElement.requestFullscreen()
                  }
                }}
                className="px-3 py-1 rounded bg-black/40 hover:bg-white/20"
              >
                Toggle
              </button>
            </div>

            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Screenshot</label>
              <button
                onClick={() => {
                  const canvas = canvasRef.current
                  if (canvas) {
                    const link = document.createElement("a")
                    link.download = "glsl-visualization.png"
                    link.href = canvas.toDataURL("image/png")
                    link.click()
                  }
                }}
                className="px-3 py-1 rounded bg-black/40 hover:bg-white/20"
              >
                Capture
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Drag and Drop Overlay */}
      {isDragOver && (
        <div className="absolute inset-0 bg-blue-500/30 flex items-center justify-center z-50">
          <div className="text-white text-2xl font-bold">
            Drop audio file here
          </div>
        </div>
      )}

      <a
        href="https://x.com/YoheiNishitsuji/status/1923362809569837131"
        target="_blank"
        rel="noopener noreferrer"
        className="absolute bottom-4 right-4 text-white text-sm opacity-80 hover:opacity-100 transition-opacity"
      >
        @Yohei Nishitsuji
      </a>
    </div>
  )
}

export default SnowGLSLVisualization
