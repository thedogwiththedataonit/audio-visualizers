"use client"

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

export default function StarWarsGLSLVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  // UI state
  const [showControls, setShowControls] = useState(false)
  const [showFps, setShowFps] = useState(false)
  const fpsRef = useRef<HTMLDivElement>(null)
  const lastTimeRef = useRef<number>(0)
  const frameCountRef = useRef<number>(0)

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
  const [tunnelSpeed, setTunnelSpeed] = useState(1.0)
  const [spiralIntensity, setSpiralIntensity] = useState(1.0)
  const [colorMode, setColorMode] = useState("default")
  const [smoothing, setSmoothing] = useState(0.85)

  // Audio refs
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioElementRef = useRef<HTMLAudioElement | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null)

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

  // WebGL refs
  const programRef = useRef<WebGLProgram | null>(null)
  const resolutionUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const timeUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  
  // Audio uniform locations
  const audioLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const bassLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const midLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const trebleLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const frequencyDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const waveformDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const audioIntensityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const bassInfluenceUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const midInfluenceUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const trebleInfluenceUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const tunnelSpeedUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const spiralIntensityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const colorModeUniformLocationRef = useRef<WebGLUniformLocation | null>(null)

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
    const gl = canvas.getContext("webgl2")
    if (!gl) {
      console.error("WebGL2 not supported")
      return
    }

    // Resize canvas to match window size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    // Vertex shader source
    const vertexShaderSource = `#version 300 es
      precision highp float;
      in vec4 a_position;
      void main() {
        gl_Position = a_position;
      }
    `

    // Fragment shader source - preserving the exact GLSL code structure with explanations
    const fragmentShaderSource = `#version 300 es
  precision highp float;
  out vec4 outColor;
  uniform vec2 u_resolution;  // Screen resolution (width, height)
  uniform float u_time;       // Time in seconds
  
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
  uniform float u_tunnelSpeed;       // Tunnel speed multiplier
  uniform float u_spiralIntensity;   // Spiral intensity multiplier
  uniform int u_colorMode;           // Color mode (0: default, 1: red, 2: blue, 3: rainbow)

  // Get frequency data for a normalized position (0-1)
  float getFrequency(float pos) {
    int index = int(pos * 63.0);
    return u_frequencyData[index];
  }

  // HSV to RGB color conversion function with audio reactivity
  // h: hue (0-1), s: saturation (0-1), v: value/brightness (0-1)
  // This is a compact implementation of the HSV color space conversion
  vec3 hsv(float h,float s,float v){
    // Apply audio-reactive hue modulation based on color mode
    if (u_colorMode == 1) { // Red/Orange
      h = 0.05 + u_trebleLevel * 0.1 * u_audioIntensity * u_trebleInfluence;
    } else if (u_colorMode == 2) { // Blue/Cyan
      h = 0.6 + u_midLevel * 0.1 * u_audioIntensity * u_midInfluence;
    } else if (u_colorMode == 3) { // Rainbow
      h = fract(h + u_time * 0.1 + u_audioLevel * 0.2 * u_audioIntensity);
    } else {
      // Default: yellow/orange with audio modulation
      h = 0.1 + u_audioLevel * 0.1 * u_audioIntensity;
    }
    
    // Audio-reactive saturation and brightness
    s = s + u_audioLevel * 0.3 * u_audioIntensity;
    v = v * (1.0 + u_bassLevel * u_bassLevel * 0.5 * u_audioIntensity * u_bassInfluence);
    
    vec4 t=vec4(1.,2./3.,1./3.,3.);  // Constants for HSV conversion
    vec3 p=abs(fract(vec3(h)+t.xyz)*6.-vec3(t.w));  // Color channel calculations
    return v*mix(vec3(t.x),clamp(p-vec3(t.x),0.,1.),s);  // Final RGB blend
  }

  void main() {
    vec2 r = u_resolution;                // Screen resolution
    vec2 FC = gl_FragCoord.xy;            // Current pixel coordinates
    float t = u_time;                     // Current time
    vec4 o = vec4(0,0,0,1);               // Output color (starts black, fully opaque)
    
    // Variable declarations:
    // i: loop counter for ray steps
    // e: distance estimation value
    // R: radius/length of position vector
    // s: scale factor for fractal iteration
    float i,e,R,s;
    
    // q: current ray position
    // p: transformed position for fractal evaluation
    // d: ray direction vector (normalized screen coordinates with depth component)
    vec3 q,p,d=vec3((FC.xy-r*.5)/r,.1);   // Ray direction from screen center with z=0.1
    
    // Audio-reactive iteration count
    float audioIterations = 99.0 * (1.0 + u_audioLevel * 0.3 * u_audioIntensity);
    
    // Ray marching loop - iterates with audio-reactive count to create the visual effect
    // This is essentially a distance field raymarcher
    for(q.zx--;i++<audioIterations;){
      // Get frequency data for current iteration
      float freqPos = i / audioIterations;
      float freqIntensity = getFrequency(freqPos);
      
      // Accumulate color based on distance estimation (e) and scale (s) with audio reactivity
      // This creates a glow effect where the ray gets close to the surface
      // Audio modulates the hue and intensity
      float audioHue = 0.1 + freqIntensity * 0.2 * u_audioIntensity;
      float audioSat = e + u_audioLevel * 0.2 * u_audioIntensity;
      float audioBrightness = min(e*s,.4-e)/25.0 * (1.0 + u_bassLevel * 0.5 * u_audioIntensity * u_bassInfluence);
      o.rgb+=hsv(audioHue, audioSat, audioBrightness);
      
      // Reset scale factor for the inner loop with audio modulation
      s = 2.0 + u_midLevel * 1.0 * u_audioIntensity * u_midInfluence;
      
      // Move ray forward by distance e*R*0.7 (safe step distance)
      // This is the core of the raymarching algorithm with audio-reactive step size
      float audioStepSize = 0.7 * (1.0 + u_audioLevel * 0.2 * u_audioIntensity);
      p=q+=d*e*R*audioStepSize;
      
      // Transform position into a logarithmic spiral space with audio reactivity
      // This creates the spiral tunnel effect with time-based rotation
      // Audio affects tunnel speed and spiral intensity
      float audioTunnelSpeed = t * 0.5 * u_tunnelSpeed * (1.0 + u_bassLevel * u_bassInfluence * u_audioIntensity);
      float audioSpiralRotation = t * 0.5 * u_spiralIntensity * (1.0 + u_midLevel * u_midInfluence * u_audioIntensity);
      float audioSpiralAmplitude = 0.1 * (1.0 + u_trebleLevel * u_trebleInfluence * u_audioIntensity);
      
      p=vec3(
        log2(R=length(p)) - audioTunnelSpeed,
        exp2(0.9-p.z/R) * (1.0 + u_audioLevel * 0.2 * u_audioIntensity),
        atan(p.y,p.x) + cos(audioSpiralRotation) * audioSpiralAmplitude
      );
      
      // Audio-reactive fractal noise generation through frequency doubling (octave summation)
      // This is similar to a simplified Perlin noise implementation
      // The loop doubles the frequency (s+=s) each iteration, creating multi-scale detail
      // Audio modulates the frequency scaling and amplitude
      float audioFreqLimit = 600.0 * (1.0 + u_audioLevel * 0.5 * u_audioIntensity);
      float audioNoiseAmplitude = 0.3 * (1.0 + u_midLevel * 0.4 * u_audioIntensity * u_midInfluence);
      
      for(e=--p.y;s<audioFreqLimit;s+=s)
        // Audio-reactive fractal noise generation:
        // 1. Cosine waves modulated by audio frequencies
        // 2. Interference patterns affected by bass/treble levels
        // 3. Frequency scaling influenced by mid frequencies
        // The result is noise that responds to the music's rhythm and frequency content
        e+=dot(
          cos(p.yzy*s*(1.0 + u_trebleLevel * 0.2 * u_audioIntensity * u_trebleInfluence))-.5,
          1.-cos(p*s*(1.0 + u_bassLevel * 0.3 * u_audioIntensity * u_bassInfluence))
        )/s*audioNoiseAmplitude;
    }
    
    // Final color output
    outColor = o;
  }
`

    // Create and compile shaders
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

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource)
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource)

    if (!vertexShader || !fragmentShader) return

    // Create program and link shaders
    const program = gl.createProgram()
    if (!program) {
      console.error("Failed to create program")
      return
    }

    gl.attachShader(program, vertexShader)
    gl.attachShader(program, fragmentShader)
    gl.linkProgram(program)

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("Program linking error:", gl.getProgramInfoLog(program))
      return
    }

    // Use our shader program - Moved outside render
    gl.useProgram(program)

    // Set up position buffer
    const positionBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
      gl.STATIC_DRAW,
    )

    // Set up position attribute
    const positionAttributeLocation = gl.getAttribLocation(program, "a_position")
    gl.enableVertexAttribArray(positionAttributeLocation)
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0)

    // Get uniform locations
    const resolutionUniformLocation = gl.getUniformLocation(program, "u_resolution")
    const timeUniformLocation = gl.getUniformLocation(program, "u_time")
    
    // Audio uniform locations
    const audioLevelUniformLocation = gl.getUniformLocation(program, "u_audioLevel")
    const bassLevelUniformLocation = gl.getUniformLocation(program, "u_bassLevel")
    const midLevelUniformLocation = gl.getUniformLocation(program, "u_midLevel")
    const trebleLevelUniformLocation = gl.getUniformLocation(program, "u_trebleLevel")
    const frequencyDataUniformLocation = gl.getUniformLocation(program, "u_frequencyData")
    const waveformDataUniformLocation = gl.getUniformLocation(program, "u_waveformData")
    const audioIntensityUniformLocation = gl.getUniformLocation(program, "u_audioIntensity")
    const bassInfluenceUniformLocation = gl.getUniformLocation(program, "u_bassInfluence")
    const midInfluenceUniformLocation = gl.getUniformLocation(program, "u_midInfluence")
    const trebleInfluenceUniformLocation = gl.getUniformLocation(program, "u_trebleInfluence")
    const tunnelSpeedUniformLocation = gl.getUniformLocation(program, "u_tunnelSpeed")
    const spiralIntensityUniformLocation = gl.getUniformLocation(program, "u_spiralIntensity")
    const colorModeUniformLocation = gl.getUniformLocation(program, "u_colorMode")
    
    // Store references
    programRef.current = program
    resolutionUniformLocationRef.current = resolutionUniformLocation
    timeUniformLocationRef.current = timeUniformLocation
    audioLevelUniformLocationRef.current = audioLevelUniformLocation
    bassLevelUniformLocationRef.current = bassLevelUniformLocation
    midLevelUniformLocationRef.current = midLevelUniformLocation
    trebleLevelUniformLocationRef.current = trebleLevelUniformLocation
    frequencyDataUniformLocationRef.current = frequencyDataUniformLocation
    waveformDataUniformLocationRef.current = waveformDataUniformLocation
    audioIntensityUniformLocationRef.current = audioIntensityUniformLocation
    bassInfluenceUniformLocationRef.current = bassInfluenceUniformLocation
    midInfluenceUniformLocationRef.current = midInfluenceUniformLocation
    trebleInfluenceUniformLocationRef.current = trebleInfluenceUniformLocation
    tunnelSpeedUniformLocationRef.current = tunnelSpeedUniformLocation
    spiralIntensityUniformLocationRef.current = spiralIntensityUniformLocation
    colorModeUniformLocationRef.current = colorModeUniformLocation

    // Start time for animation
    const startTime = performance.now()

    // Render function
    const render = () => {
      // Update time uniform
      const currentTime = (performance.now() - startTime) / 1000

      // Set viewport and clear
      gl.viewport(0, 0, canvas.width, canvas.height)
      gl.clearColor(0, 0, 0, 1)
      gl.clear(gl.COLOR_BUFFER_BIT)

      // Get current audio data
      const audioData = analyzeAudio()

      // Update basic uniforms
      if (resolutionUniformLocationRef.current) {
        gl.uniform2f(resolutionUniformLocationRef.current, canvas.width, canvas.height)
      }
      if (timeUniformLocationRef.current) {
        gl.uniform1f(timeUniformLocationRef.current, currentTime)
      }

      // Update audio uniforms
      if (audioLevelUniformLocationRef.current) {
        gl.uniform1f(audioLevelUniformLocationRef.current, audioData.level)
      }
      if (bassLevelUniformLocationRef.current) {
        gl.uniform1f(bassLevelUniformLocationRef.current, audioData.bassLevel)
      }
      if (midLevelUniformLocationRef.current) {
        gl.uniform1f(midLevelUniformLocationRef.current, audioData.midLevel)
      }
      if (trebleLevelUniformLocationRef.current) {
        gl.uniform1f(trebleLevelUniformLocationRef.current, audioData.trebleLevel)
      }
      if (frequencyDataUniformLocationRef.current) {
        gl.uniform1fv(frequencyDataUniformLocationRef.current, audioData.frequencyData)
      }
      if (waveformDataUniformLocationRef.current) {
        gl.uniform1fv(waveformDataUniformLocationRef.current, audioData.waveformData)
      }
      if (audioIntensityUniformLocationRef.current) {
        gl.uniform1f(audioIntensityUniformLocationRef.current, audioIntensity)
      }
      if (bassInfluenceUniformLocationRef.current) {
        gl.uniform1f(bassInfluenceUniformLocationRef.current, bassInfluence)
      }
      if (midInfluenceUniformLocationRef.current) {
        gl.uniform1f(midInfluenceUniformLocationRef.current, midInfluence)
      }
      if (trebleInfluenceUniformLocationRef.current) {
        gl.uniform1f(trebleInfluenceUniformLocationRef.current, trebleInfluence)
      }
      if (tunnelSpeedUniformLocationRef.current) {
        gl.uniform1f(tunnelSpeedUniformLocationRef.current, tunnelSpeed)
      }
      if (spiralIntensityUniformLocationRef.current) {
        gl.uniform1f(spiralIntensityUniformLocationRef.current, spiralIntensity)
      }

      // Set color mode
      let colorModeValue = 0
      if (colorMode === "red") colorModeValue = 1
      else if (colorMode === "blue") colorModeValue = 2
      else if (colorMode === "rainbow") colorModeValue = 3
      if (colorModeUniformLocationRef.current) {
        gl.uniform1i(colorModeUniformLocationRef.current, colorModeValue)
      }

      // Draw
      gl.drawArrays(gl.TRIANGLES, 0, 6)

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
      gl.deleteProgram(program)
      gl.deleteShader(vertexShader)
      gl.deleteShader(fragmentShader)
      gl.deleteBuffer(positionBuffer)
      gl.useProgram(null)
    }
  }, [
    analyzeAudio,
    audioIntensity,
    bassInfluence,
    midInfluence,
    trebleInfluence,
    tunnelSpeed,
    spiralIntensity,
    colorMode,
    showFps,
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
      <canvas
        ref={canvasRef}
        style={{
          display: "block",
          width: "100%",
          height: "100vh",
          position: "absolute",
          top: 0,
          left: 0,
        }}
      />

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
            {/* Color Mode */}
            <div className="space-y-2">
              <label className="block text-sm font-medium">Color Mode</label>
              <Select value={colorMode} onValueChange={setColorMode}>
                <SelectTrigger className="bg-black/50 border-gray-700">
                  <SelectValue placeholder="Select color mode" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700 text-white">
                  <SelectItem value="default">Default (Yellow/Orange)</SelectItem>
                  <SelectItem value="red">Red/Orange</SelectItem>
                  <SelectItem value="blue">Blue/Cyan</SelectItem>
                  <SelectItem value="rainbow">Rainbow</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Tunnel Speed */}
            <div className="space-y-2">
              <label className="block text-sm font-medium">Tunnel Speed: {tunnelSpeed.toFixed(1)}x</label>
              <Slider
                value={[tunnelSpeed]}
                min={0.1}
                max={3}
                step={0.1}
                onValueChange={(values) => setTunnelSpeed(values[0])}
                className="py-2"
              />
            </div>

            {/* Spiral Intensity */}
            <div className="space-y-2">
              <label className="block text-sm font-medium">Spiral Intensity: {spiralIntensity.toFixed(1)}x</label>
              <Slider
                value={[spiralIntensity]}
                min={0.1}
                max={3}
                step={0.1}
                onValueChange={(values) => setSpiralIntensity(values[0])}
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
                    link.download = "starwars-glsl-visualization.png"
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
        href="https://x.com/YoheiNishitsuji/status/1915756430084256054"
        target="_blank"
        rel="noopener noreferrer"
        className="absolute bottom-4 right-4 text-white text-sm opacity-70 hover:opacity-100 transition-opacity z-10"
      >
        @Yohei Nishitsuji
      </a>
    </div>
  )
}
