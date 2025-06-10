"use client"

import { useRef, useEffect, useState, useCallback } from "react"
import ColorThemeSelector from "../ColorThemeSelector"

interface AudioData {
  level: number
  bassLevel: number
  midLevel: number
  trebleLevel: number
  frequencyData: Float32Array
  waveformData: Float32Array
}

export default function FractalGLSLVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)
  const startTimeRef = useRef<number>(0)

  // Store all WebGL related objects in refs to avoid re-renders
  const glRef = useRef<WebGL2RenderingContext | null>(null)
  const programRef = useRef<WebGLProgram | null>(null)
  const uniformLocationsRef = useRef<{
    resolution: WebGLUniformLocation | null
    time: WebGLUniformLocation | null
    mouse: WebGLUniformLocation | null
    speed: WebGLUniformLocation | null
    intensity: WebGLUniformLocation | null
    baseHue: WebGLUniformLocation | null
    saturation: WebGLUniformLocation | null
    audioLevel: WebGLUniformLocation | null
    bassLevel: WebGLUniformLocation | null
    midLevel: WebGLUniformLocation | null
    trebleLevel: WebGLUniformLocation | null
    frequencyData: WebGLUniformLocation | null
    waveformData: WebGLUniformLocation | null
    colorSensitivity: WebGLUniformLocation | null
    beatPulse: WebGLUniformLocation | null
    scaleReactivity: WebGLUniformLocation | null
    rotationSpeed: WebGLUniformLocation | null
    fractalComplexity: WebGLUniformLocation | null
  }>({
    resolution: null,
    time: null,
    mouse: null,
    speed: null,
    intensity: null,
    baseHue: null,
    saturation: null,
    audioLevel: null,
    bassLevel: null,
    midLevel: null,
    trebleLevel: null,
    frequencyData: null,
    waveformData: null,
    colorSensitivity: null,
    beatPulse: null,
    scaleReactivity: null,
    rotationSpeed: null,
    fractalComplexity: null,
  })

  // Audio processing refs
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioElementRef = useRef<HTMLAudioElement | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null)

  // State for UI controls
  const [mousePosition, setMousePosition] = useState<[number, number]>([0.5, 0.5])
  const [speed, setSpeed] = useState<number>(1.0)
  const [intensity, setIntensity] = useState<number>(1.0)
  const [showControls, setShowControls] = useState<boolean>(true)
  const [baseHue, setBaseHue] = useState<number>(0.08)
  const [saturation, setSaturation] = useState<number>(0.8)
  const [activeTab, setActiveTab] = useState<"controls" | "themes" | "audio">("controls")

  // Audio state
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.7)
  const [trackName, setTrackName] = useState("")
  const [isDragOver, setIsDragOver] = useState(false)

  // Audio-reactive control state
  const [colorSensitivity, setColorSensitivity] = useState<number>(0.5)
  const [beatPulse, setBeatPulse] = useState<number>(0.5)
  const [smoothing, setSmoothing] = useState<number>(0.85)
  const [scaleReactivity, setScaleReactivity] = useState<number>(0.5)
  const [rotationSpeed, setRotationSpeed] = useState<number>(0.5)
  const [fractalComplexity, setFractalComplexity] = useState<number>(0.5)

  // Store current values in refs so the animation loop can access them without dependencies
  const mousePositionRef = useRef<[number, number]>([0.5, 0.5])
  const speedRef = useRef<number>(1.0)
  const intensityRef = useRef<number>(1.0)
  const baseHueRef = useRef<number>(0.08)
  const saturationRef = useRef<number>(0.8)
  const colorSensitivityRef = useRef<number>(0.5)
  const beatPulseRef = useRef<number>(0.5)
  const scaleReactivityRef = useRef<number>(0.5)
  const rotationSpeedRef = useRef<number>(0.5)
  const fractalComplexityRef = useRef<number>(0.5)

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

  // Update refs when state changes
  useEffect(() => {
    mousePositionRef.current = mousePosition
  }, [mousePosition])

  useEffect(() => {
    speedRef.current = speed
  }, [speed])

  useEffect(() => {
    intensityRef.current = intensity
  }, [intensity])

  useEffect(() => {
    baseHueRef.current = baseHue
  }, [baseHue])

  useEffect(() => {
    saturationRef.current = saturation
  }, [saturation])

  useEffect(() => {
    colorSensitivityRef.current = colorSensitivity
  }, [colorSensitivity])

  useEffect(() => {
    beatPulseRef.current = beatPulse
  }, [beatPulse])

  useEffect(() => {
    scaleReactivityRef.current = scaleReactivity
  }, [scaleReactivity])

  useEffect(() => {
    rotationSpeedRef.current = rotationSpeed
  }, [rotationSpeed])

  useEffect(() => {
    fractalComplexityRef.current = fractalComplexity
  }, [fractalComplexity])

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

  // Toggle fullscreen function
  const toggleFullscreen = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    if (!document.fullscreenElement) {
      canvas.requestFullscreen().catch((err) => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`)
      })
    } else {
      document.exitFullscreen()
    }
  }

  const render = useCallback((timestamp: number) => {
    const gl = glRef.current
    const program = programRef.current
    const uniforms = uniformLocationsRef.current
    const canvas = canvasRef.current

    if (!gl || !program || !canvas) return

    // Calculate elapsed time in seconds
    const elapsedTime = (timestamp - startTimeRef.current) / 1000

    // Get current audio data
    const audioData = analyzeAudio()

    // Update uniforms
    gl.uniform2f(uniforms.resolution!, canvas.width, canvas.height)
    gl.uniform1f(uniforms.time!, elapsedTime)
    gl.uniform2f(uniforms.mouse!, mousePositionRef.current[0], mousePositionRef.current[1])
    gl.uniform1f(uniforms.speed!, speedRef.current)
    gl.uniform1f(uniforms.intensity!, intensityRef.current)
    gl.uniform1f(uniforms.baseHue!, baseHueRef.current)
    gl.uniform1f(uniforms.saturation!, saturationRef.current)

    // Update audio uniforms
    gl.uniform1f(uniforms.audioLevel!, audioData.level)
    gl.uniform1f(uniforms.bassLevel!, audioData.bassLevel)
    gl.uniform1f(uniforms.midLevel!, audioData.midLevel)
    gl.uniform1f(uniforms.trebleLevel!, audioData.trebleLevel)
    gl.uniform1fv(uniforms.frequencyData!, audioData.frequencyData)
    gl.uniform1fv(uniforms.waveformData!, audioData.waveformData)
    gl.uniform1f(uniforms.colorSensitivity!, colorSensitivityRef.current)
    gl.uniform1f(uniforms.beatPulse!, beatPulseRef.current)
    gl.uniform1f(uniforms.scaleReactivity!, scaleReactivityRef.current)
    gl.uniform1f(uniforms.rotationSpeed!, rotationSpeedRef.current)
    gl.uniform1f(uniforms.fractalComplexity!, fractalComplexityRef.current)

    // Clear and draw
    gl.clearColor(0, 0, 0, 1)
    gl.clear(gl.COLOR_BUFFER_BIT)
    gl.drawArrays(gl.TRIANGLES, 0, 6)

    // Continue animation loop
    animationRef.current = requestAnimationFrame(render)
  }, [analyzeAudio])

  // Setup WebGL and start animation
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // Get WebGL2 context
    const gl = canvas.getContext("webgl2")
    if (!gl) {
      console.error("WebGL2 not supported")
      return
    }
    glRef.current = gl

    // Resize canvas to fill window
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
      gl.viewport(0, 0, canvas.width, canvas.height)
    }
    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    // Vertex shader - simple pass-through
    const vertexShaderSource = `#version 300 es
      in vec4 a_position;
      void main() {
        gl_Position = a_position;
      }
    `

    // Enhanced audio-reactive fragment shader
    const fragmentShaderSource = `#version 300 es
  precision highp float;
  out vec4 outColor;
  uniform vec2 u_resolution;
  uniform vec2 u_mouse;
  uniform float u_time;
  uniform float u_speed;
  uniform float u_intensity;
  uniform float u_baseHue;
  uniform float u_saturation;
  
  // Audio uniforms
  uniform float u_audioLevel;
  uniform float u_bassLevel;
  uniform float u_midLevel;
  uniform float u_trebleLevel;
  uniform float u_frequencyData[64];
  uniform float u_waveformData[32];
  uniform float u_colorSensitivity;
  uniform float u_beatPulse;
  uniform float u_scaleReactivity;
  uniform float u_rotationSpeed;
  uniform float u_fractalComplexity;
  
  // Rotation function (needed for rotate3D in the original code)
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
  
  // HSV to RGB conversion
  vec3 hsv(float h, float s, float v) {
    vec4 t = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + t.xyz) * 6.0 - vec3(t.w));
    return v * mix(vec3(t.x), clamp(p - vec3(t.x), 0.0, 1.0), s);
  }
  
  // Get frequency data for a normalized position (0-1)
  float getFrequency(float pos) {
    int index = int(pos * 63.0);
    return u_frequencyData[index];
  }
  
  void main() {
    vec2 r = u_resolution;
    vec2 FC = gl_FragCoord.xy;
    float t = u_time * u_speed;
    vec4 o = vec4(0, 0, 0, 1);
    
    // Audio-reactive scaling - replace static breathing with audio level
    float audioScale = 1.0 + (u_audioLevel * 0.5 + u_bassLevel * 0.3) * u_scaleReactivity;
    
    // Use mouse position to influence rotation axis, with treble affecting Z component
    vec3 rotAxis = normalize(vec3(u_mouse.x, u_mouse.y, 0.5 + u_trebleLevel * 0.3));
    
    // Dynamic iteration count based on audio intensity and fractal complexity control
    float maxIterations = 18.0 + u_audioLevel * 6.0 * u_fractalComplexity;
    
    for(float i=0.,g=0.,e=0.,s=0.; i < maxIterations; i++){
      // Audio-reactive 3D positioning with scale
      vec3 p=vec3((FC.xy-.5*r)/r.y*3.5*audioScale,g+.5);
      
      // Audio-reactive rotation - bass affects main rotation, treble adds secondary rotation
      float rotSpeed = t * (0.5 + u_bassLevel * 0.8 * u_intensity) * u_rotationSpeed;
      p = p * rotate3D(rotSpeed, rotAxis);
      
      // Additional treble-based rotation on different axis
      vec3 trebleAxis = normalize(vec3(0.7, 0.3, u_trebleLevel));
      p = p * rotate3D(t * u_trebleLevel * 0.4 * u_rotationSpeed, trebleAxis);
      
      s=1.;
      
      // Audio-reactive folding parameters
      float bassBoost = 1.0 + u_bassLevel * 0.5 * u_intensity;
      float midBoost = 1.0 + u_midLevel * 0.3 * u_intensity;
      float trebleBoost = 1.0 + u_trebleLevel * 0.2 * u_intensity;
      
      vec3 foldOffset = vec3(2.2 * bassBoost, 3.0 * midBoost, 3.0 * trebleBoost);
      vec3 foldScale = vec3(0, 3.01 * (1.0 + u_audioLevel * 0.1), 3);
      
      for(int i=0;i++<40;p=foldScale-abs(abs(p)*e-foldOffset)) {
        // Audio-reactive scale factor
        float dotProduct = dot(p,p);
        float audioInfluence = 10.0 + u_audioLevel * 5.0 * u_intensity;
        s*=e=max(1., audioInfluence/dotProduct);
      }
      
      // Audio-reactive geometric accumulation
      float geoMod = mod(length(p.yy-p.xy*.3),p.y);
      float audioGeoBoost = 0.4 * (1.0 + u_audioLevel * 0.2 * u_intensity);
      g-=geoMod/s*audioGeoBoost;
      
      // Dynamic color based on frequency spectrum and position
      float freqPos = i / maxIterations;
      float freqIntensity = getFrequency(freqPos);
      
      // Multi-layered hue calculation with audio reactivity
      float baseHueShift = u_baseHue + p.x * 0.1;
      float bassHueShift = u_bassLevel * 0.15 * u_colorSensitivity;
      float midHueShift = u_midLevel * 0.1 * u_colorSensitivity;
      float trebleHueShift = u_trebleLevel * 0.2 * u_colorSensitivity;
      float freqHueShift = freqIntensity * 0.05 * u_colorSensitivity;
      
      float finalHue = baseHueShift + bassHueShift + midHueShift + trebleHueShift + freqHueShift;
      
      // Audio-reactive saturation
      float finalSaturation = u_saturation + 0.3*p.x + u_audioLevel * 0.1 * u_colorSensitivity;
      
      // Beat-reactive brightness with pulse effect
      float beatPulseEffect = 1.0 + u_bassLevel * u_bassLevel * 0.5 * u_beatPulse;
      float brightness = (s/4e3 * u_intensity) * beatPulseEffect * (1.0 + freqIntensity * 0.2);
      
      o.rgb+=hsv(finalHue, finalSaturation, brightness);
    }
    
    // Final audio-reactive enhancement
    o.rgb *= 1.0 + u_audioLevel * 0.1 * u_intensity;
    
    // Add subtle waveform visualization in corner
    vec2 cornerPos = FC.xy / r;
    if (cornerPos.x < 0.08 && cornerPos.y > 0.92) {
      float waveIndex = cornerPos.x * 12.5; // 0.08 * 12.5 = 1.0
      int waveIdx = int(waveIndex * 31.0);
      if (waveIdx >= 0 && waveIdx < 32) {
        float waveValue = u_waveformData[waveIdx];
        o.rgb += vec3(0.1, 0.1, 0.2) * abs(waveValue) * 2.0;
      }
    }
    
    outColor = o;
  }
`

    // Create shader program
    const createShader = (gl: WebGL2RenderingContext, type: number, source: string) => {
      const shader = gl.createShader(type)
      if (!shader) {
        console.error("Failed to create shader")
        return null
      }
      gl.shaderSource(shader, source)
      gl.compileShader(shader)

      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error(`Shader compilation error: ${gl.getShaderInfoLog(shader)}`)
        gl.deleteShader(shader)
        return null
      }

      return shader
    }

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource)
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource)

    if (!vertexShader || !fragmentShader) return

    const program = gl.createProgram()
    if (!program) {
      console.error("Failed to create program")
      return
    }

    gl.attachShader(program, vertexShader)
    gl.attachShader(program, fragmentShader)
    gl.linkProgram(program)

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(`Program linking error: ${gl.getProgramInfoLog(program)}`)
      return
    }

    programRef.current = program

    // Set up position buffer (full screen quad)
    const positionBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
    const positions = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW)

    // Set up position attribute
    const positionAttributeLocation = gl.getAttribLocation(program, "a_position")
    gl.enableVertexAttribArray(positionAttributeLocation)
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0)

    // Set up uniforms
    uniformLocationsRef.current = {
      resolution: gl.getUniformLocation(program, "u_resolution"),
      time: gl.getUniformLocation(program, "u_time"),
      mouse: gl.getUniformLocation(program, "u_mouse"),
      speed: gl.getUniformLocation(program, "u_speed"),
      intensity: gl.getUniformLocation(program, "u_intensity"),
      baseHue: gl.getUniformLocation(program, "u_baseHue"),
      saturation: gl.getUniformLocation(program, "u_saturation"),
      audioLevel: gl.getUniformLocation(program, "u_audioLevel"),
      bassLevel: gl.getUniformLocation(program, "u_bassLevel"),
      midLevel: gl.getUniformLocation(program, "u_midLevel"),
      trebleLevel: gl.getUniformLocation(program, "u_trebleLevel"),
      frequencyData: gl.getUniformLocation(program, "u_frequencyData"),
      waveformData: gl.getUniformLocation(program, "u_waveformData"),
      colorSensitivity: gl.getUniformLocation(program, "u_colorSensitivity"),
      beatPulse: gl.getUniformLocation(program, "u_beatPulse"),
      scaleReactivity: gl.getUniformLocation(program, "u_scaleReactivity"),
      rotationSpeed: gl.getUniformLocation(program, "u_rotationSpeed"),
      fractalComplexity: gl.getUniformLocation(program, "u_fractalComplexity"),
    }

    // Add mouse event handlers
    const handleMouseMove = (event: MouseEvent) => {
      const canvas = canvasRef.current
      if (!canvas) return

      const rect = canvas.getBoundingClientRect()
      const x = (event.clientX - rect.left) / canvas.width
      const y = 1.0 - (event.clientY - rect.top) / canvas.height // Flip Y for WebGL
      setMousePosition([x, y])
    }

    window.addEventListener("mousemove", handleMouseMove)

    // Add double-click event listener
    canvas.addEventListener("dblclick", toggleFullscreen)

    // Set start time
    startTimeRef.current = performance.now()

    // Start the animation loop
    animationRef.current = requestAnimationFrame(render)

    // Cleanup
    return () => {
      window.removeEventListener("resize", resizeCanvas)
      window.removeEventListener("mousemove", handleMouseMove)
      canvas.removeEventListener("dblclick", toggleFullscreen)
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current)
      }
      if (gl && program) {
        gl.deleteProgram(program)
      }
      if (gl && vertexShader) {
        gl.deleteShader(vertexShader)
      }
      if (gl && fragmentShader) {
        gl.deleteShader(fragmentShader)
      }
      if (gl && positionBuffer) {
        gl.deleteBuffer(positionBuffer)
      }
      if (gl) {
        gl.useProgram(null)
      }
    }
  }, [render]) // Empty dependency array - only run once on mount

  useEffect(() => {
    if (glRef.current && programRef.current) {
      glRef.current.useProgram(programRef.current)
    }
  }, [])

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

  // Update the return statement to include audio controls
  return (
    <div 
      className="relative w-full h-screen"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <canvas ref={canvasRef} className="w-full h-screen" />

      {/* Audio Controls Panel - Top Left */}
      <div
        className="absolute top-4 left-4 bg-black/70 p-4 rounded-lg text-white backdrop-blur-sm border border-white/10"
        style={{ minWidth: "280px", zIndex: 10 }}
      >
        {/* File Upload */}
        <div className="mb-3">
          <label className="block mb-2 text-xs opacity-80">Upload Audio File</label>
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            className="w-full p-2 bg-white/10 border border-white/30 rounded text-white text-xs"
          />
        </div>

        {/* Default Songs */}
        <div className="mb-3">
          <label className="block mb-2 text-xs opacity-80">Or Choose a Default Song</label>
          <div className="space-y-1">
            {defaultSongs.map((song, index) => (
              <button
                key={index}
                onClick={() => loadDefaultSong(song.filename, song.displayName)}
                className="w-full p-2 bg-white/10 hover:bg-white/20 border border-white/30 rounded text-white text-xs text-left transition-colors"
              >
                {song.displayName}
              </button>
            ))}
          </div>
        </div>

        {/* Track Info */}
        {trackName && (
          <div className="mb-3 text-sm font-bold truncate" title={trackName}>
            {trackName}
          </div>
        )}

        {/* Play/Pause Button */}
        <div className="mb-3 text-center">
          <button
            onClick={togglePlayPause}
            disabled={!audioElementRef.current}
            className={`px-4 py-2 rounded-full text-white font-bold text-sm cursor-pointer transition-colors ${
              isPlaying ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600"
            } ${!audioElementRef.current ? "opacity-50 cursor-not-allowed" : ""}`}
          >
            {isPlaying ? "⏸ Pause" : "▶ Play"}
          </button>
        </div>

        {/* Timeline */}
        {duration > 0 && (
          <div className="mb-3">
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
              className="w-full h-1 bg-white/30 rounded outline-none"
            />
          </div>
        )}

        {/* Volume Control */}
        <div className="mb-3">
          <label className="block mb-1 text-xs opacity-80">
            Volume: {Math.round(volume * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={volume * 100}
            onChange={handleVolumeChange}
            className="w-full h-1 bg-white/30 rounded outline-none"
          />
        </div>

        {/* Visual Controls Divider */}
        <div className="border-t border-white/20 pt-3 mb-3">
          <div className="text-xs font-bold opacity-90 mb-2">Visual Controls</div>
          
          {/* Intensity */}
          <div className="mb-2">
            <label className="block mb-1 text-xs opacity-80">
              Intensity: {Math.round(intensity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="300"
              value={intensity * 100}
              onChange={(e) => setIntensity(parseFloat(e.target.value) / 100)}
              className="w-full h-1 bg-white/30 rounded outline-none"
            />
          </div>

          {/* Rotation Speed */}
          <div className="mb-2">
            <label className="block mb-1 text-xs opacity-80">
              Rotation Speed: {Math.round(rotationSpeed * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={rotationSpeed * 100}
              onChange={(e) => setRotationSpeed(parseFloat(e.target.value) / 100)}
              className="w-full h-1 bg-white/30 rounded outline-none"
            />
          </div>

          {/* Color Sensitivity */}
          <div className="mb-2">
            <label className="block mb-1 text-xs opacity-80">
              Color Sensitivity: {Math.round(colorSensitivity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={colorSensitivity * 100}
              onChange={(e) => setColorSensitivity(parseFloat(e.target.value) / 100)}
              className="w-full h-1 bg-white/30 rounded outline-none"
            />
          </div>

          {/* Beat Pulse */}
          <div className="mb-2">
            <label className="block mb-1 text-xs opacity-80">
              Beat Pulse: {Math.round(beatPulse * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={beatPulse * 100}
              onChange={(e) => setBeatPulse(parseFloat(e.target.value) / 100)}
              className="w-full h-1 bg-white/30 rounded outline-none"
            />
          </div>

          {/* Fractal Complexity */}
          <div className="mb-2">
            <label className="block mb-1 text-xs opacity-80">
              Fractal Complexity: {Math.round(fractalComplexity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="150"
              value={fractalComplexity * 100}
              onChange={(e) => setFractalComplexity(parseFloat(e.target.value) / 100)}
              className="w-full h-1 bg-white/30 rounded outline-none"
            />
          </div>

          {/* Scale Reactivity */}
          <div className="mb-2">
            <label className="block mb-1 text-xs opacity-80">
              Scale Reactivity: {Math.round(scaleReactivity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={scaleReactivity * 100}
              onChange={(e) => setScaleReactivity(parseFloat(e.target.value) / 100)}
              className="w-full h-1 bg-white/30 rounded outline-none"
            />
          </div>

          {/* Smoothing
          <div className="mb-2">
            <label className="block mb-1 text-xs opacity-80">
              Smoothing: {Math.round(smoothing * 100)}%
            </label>
            <input
              type="range"
              min="10"
              max="95"
              value={smoothing * 100}
              onChange={(e) => setSmoothing(parseFloat(e.target.value) / 100)}
              className="w-full h-1 bg-white/30 rounded outline-none"
            />
          </div>
           */}

          {/* Reset Button */}
          <div className="text-center mt-3">
            <button
              onClick={() => {
                setIntensity(1.0)
                setRotationSpeed(0.5)
                setColorSensitivity(0.5)
                setBeatPulse(0.5)
                setFractalComplexity(0.5)
                setScaleReactivity(0.5)
                setSmoothing(0.85)
              }}
              className="px-3 py-1 bg-white/20 hover:bg-white/30 rounded text-xs"
            >
              Reset Controls
            </button>
          </div>
        </div>

        {/* Audio Levels Display */}
        {isPlaying && (
          <div className="text-xs opacity-70 border-t border-white/20 pt-2">
            <div>Level: {Math.round(smoothedAudioDataRef.current.level * 100)}%</div>
            <div>Bass: {Math.round(smoothedAudioDataRef.current.bassLevel * 100)}%</div>
            <div>Mid: {Math.round(smoothedAudioDataRef.current.midLevel * 100)}%</div>
            <div>Treble: {Math.round(smoothedAudioDataRef.current.trebleLevel * 100)}%</div>
          </div>
        )}
      </div>

      {/* Main Controls Panel - Top Right */}
      {showControls && (
        <div
          className="absolute top-4 right-4 bg-black/70 p-4 rounded-lg text-white backdrop-blur-sm border border-white/10"
          style={{ width: activeTab === "themes" ? "320px" : "auto" }}
        >
          <div className="flex justify-between items-center mb-4">
            <div className="flex gap-2">
              <button
                onClick={() => setActiveTab("controls")}
                className={`px-3 py-1 rounded-md text-sm ${activeTab === "controls" ? "bg-white/20" : "hover:bg-white/10"}`}
              >
                Controls
              </button>
              <button
                onClick={() => setActiveTab("audio")}
                className={`px-3 py-1 rounded-md text-sm ${activeTab === "audio" ? "bg-white/20" : "hover:bg-white/10"}`}
              >
                Audio FX
              </button>
              <button
                onClick={() => setActiveTab("themes")}
                className={`px-3 py-1 rounded-md text-sm ${activeTab === "themes" ? "bg-white/20" : "hover:bg-white/10"}`}
              >
                Themes
              </button>
            </div>
            <button onClick={() => setShowControls(false)} className="text-white hover:text-gray-300 ml-2">
              Hide
            </button>
          </div>

          {activeTab === "controls" && (
            <>
              <div className="mb-4">
                <label className="block mb-2">Speed: {speed.toFixed(1)}</label>
                <input
                  type="range"
                  min="0.1"
                  max="3.0"
                  step="0.1"
                  value={speed}
                  onChange={(e) => setSpeed(Number.parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="mb-4">
                <label className="block mb-2">Intensity: {intensity.toFixed(1)}</label>
                <input
                  type="range"
                  min="0.1"
                  max="3.0"
                  step="0.1"
                  value={intensity}
                  onChange={(e) => setIntensity(Number.parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="mb-4">
                <button onClick={toggleFullscreen} className="w-full bg-white/20 hover:bg-white/30 py-2 rounded-md">
                  Toggle Fullscreen
                </button>
              </div>

              <div className="text-sm opacity-70 mt-4">
                <p>Move your mouse to change rotation</p>
                <p>Double-click for fullscreen</p>
                <p>Upload audio for reactive effects</p>
              </div>
            </>
          )}

          {activeTab === "audio" && (
            <>
              <div className="mb-3">
                <label className="block mb-1 text-sm">Color Sensitivity: {Math.round(colorSensitivity * 100)}%</label>
                <input
                  type="range"
                  min="0"
                  max="200"
                  value={colorSensitivity * 100}
                  onChange={(e) => setColorSensitivity(parseFloat(e.target.value) / 100)}
                  className="w-full"
                />
              </div>

              <div className="mb-3">
                <label className="block mb-1 text-sm">Beat Pulse: {Math.round(beatPulse * 100)}%</label>
                <input
                  type="range"
                  min="0"
                  max="200"
                  value={beatPulse * 100}
                  onChange={(e) => setBeatPulse(parseFloat(e.target.value) / 100)}
                  className="w-full"
                />
              </div>

              <div className="mb-3">
                <label className="block mb-1 text-sm">Scale Reactivity: {Math.round(scaleReactivity * 100)}%</label>
                <input
                  type="range"
                  min="0"
                  max="200"
                  value={scaleReactivity * 100}
                  onChange={(e) => setScaleReactivity(parseFloat(e.target.value) / 100)}
                  className="w-full"
                />
              </div>

              <div className="mb-3">
                <label className="block mb-1 text-sm">Smoothing: {Math.round(smoothing * 100)}%</label>
                <input
                  type="range"
                  min="10"
                  max="95"
                  value={smoothing * 100}
                  onChange={(e) => setSmoothing(parseFloat(e.target.value) / 100)}
                  className="w-full"
                />
              </div>

              <div className="text-center mt-4">
                <button
                  onClick={() => {
                    setIntensity(1.0)
                    setRotationSpeed(0.5)
                    setColorSensitivity(0.5)
                    setBeatPulse(0.5)
                    setFractalComplexity(0.5)
                    setScaleReactivity(0.5)
                    setSmoothing(0.85)
                  }}
                  className="px-3 py-1 bg-white/20 hover:bg-white/30 rounded text-sm"
                >
                  Reset Audio FX
                </button>
              </div>

              <div className="text-xs opacity-70 mt-4">
                <p>• Color Sensitivity: How much audio affects colors</p>
                <p>• Beat Pulse: Brightness response to bass hits</p>
                <p>• Scale Reactivity: Size changes with audio</p>
                <p>• Smoothing: Reduces jittery movements</p>
              </div>
            </>
          )}

          {activeTab === "themes" && (
            <div className="max-h-[70vh] overflow-y-auto pr-1">
              <ColorThemeSelector
                onSelectTheme={(theme) => {
                  setBaseHue(theme.baseHue)
                  setSaturation(theme.saturation)
                }}
              />
            </div>
          )}
        </div>
      )}

      {!showControls && (
        <button
          onClick={() => setShowControls(true)}
          className="absolute top-4 right-4 bg-black/70 p-2 rounded-lg text-white hover:bg-black/90"
        >
          Show Controls
        </button>
      )}

      {/* Drag and Drop Overlay */}
      {isDragOver && (
        <div className="absolute inset-0 bg-blue-500/20 flex items-center justify-center z-20 text-white text-2xl font-bold">
          Drop audio file here
        </div>
      )}
    </div>
  )
}
