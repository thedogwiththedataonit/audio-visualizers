"use client"

import type React from "react"
import { useRef, useEffect, useState, useCallback } from "react"

interface AudioData {
  level: number
  bassLevel: number
  midLevel: number
  trebleLevel: number
  frequencyData: Float32Array
  waveformData: Float32Array
}

/**
 * Audio-Reactive Tornado GLSL Visualization Component
 *
 * An enhanced React component that renders a WebGL tornado visualization based on ray marching
 * with comprehensive audio reactivity and visual controls.
 *
 * Features:
 * - Audio-reactive tornado rendering with frequency analysis
 * - Full audio player with upload, playback controls, and timeline
 * - Real-time visual controls for audio responsiveness
 * - Mouse inertia controls for manual interaction
 * - Support for MP3, WAV, and OGG audio formats
 */
const TornadoGLSLVisualization: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const programRef = useRef<WebGLProgram | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioElementRef = useRef<HTMLAudioElement | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null)
  const animationRef = useRef<number>(0)

  // WebGL uniform locations
  const resolutionUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const timeUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const mouseUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const rotationXUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const rotationYUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const colorShiftUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  
  // Audio uniform locations
  const audioLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const bassLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const midLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const trebleLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const frequencyDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const waveformDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  
  // Audio control uniform locations
  const intensityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const rotationSpeedUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const colorSensitivityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const beatPulseUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const fractalComplexityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const scaleReactivityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)

  // Audio state
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.7)
  const [trackName, setTrackName] = useState("")
  const [isDragOver, setIsDragOver] = useState(false)

  // Visual control state
  const [intensity, setIntensity] = useState(0.5)
  const [rotationSpeed, setRotationSpeed] = useState(0.5)
  const [colorSensitivity, setColorSensitivity] = useState(0.5)
  const [beatPulse, setBeatPulse] = useState(0.5)
  const [fractalComplexity, setFractalComplexity] = useState(0.5)
  const [smoothing, setSmoothing] = useState(0.85)
  const [scaleReactivity, setScaleReactivity] = useState(0.5)

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
    let program: WebGLProgram | null = null
    let vertexShader: WebGLShader | null = null
    let fragmentShader: WebGLShader | null = null
    let positionBuffer: WebGLBuffer | null = null
    let gl: WebGL2RenderingContext | null = null
    const canvas = canvasRef.current
    if (!canvas) return

    // Get WebGL2 context
    gl = canvas.getContext("webgl2")
    if (!gl) {
      console.error("WebGL2 not supported")
      return
    }

    // Resize canvas to full screen
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
      gl.viewport(0, 0, canvas.width, canvas.height)
    }

    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    // Vertex shader source - minimal pass-through shader
    const vertexShaderSource = `#version 300 es
    precision highp float;
    
    in vec4 a_position;
    
    void main() {
      gl_Position = a_position;
    }
    `

    // Audio-reactive tornado fragment shader
    const fragmentShaderSource = `#version 300 es
precision highp float;

out vec4 outColor;
uniform vec2 u_resolution;
uniform float u_time;

// Audio uniforms
uniform float u_audioLevel;
uniform float u_bassLevel;
uniform float u_midLevel;
uniform float u_trebleLevel;
uniform float u_frequencyData[64];
uniform float u_waveformData[32];

// Control uniforms
uniform float u_intensity;
uniform float u_rotationSpeed;
uniform float u_colorSensitivity;
uniform float u_beatPulse;
uniform float u_fractalComplexity;
uniform float u_scaleReactivity;

// HSV to RGB conversion function
vec3 hsv(float h, float s, float v) {
  vec4 t = vec4(1., 2./3., 1./3., 3.);
  vec3 p = abs(fract(vec3(h) + t.xyz) * 6. - vec3(t.w));
  return v * mix(vec3(t.x), clamp(p - vec3(t.x), 0., 1.), s);
}

// 2D rotation matrix
mat2 r(float a) {
  return mat2(cos(a), -sin(a), sin(a), cos(a));
}

// Get frequency data for a normalized position (0-1)
float getFrequency(float pos) {
  int index = int(pos * 63.0);
  return u_frequencyData[index];
}

void main() {
  vec2 FC = gl_FragCoord.xy;
  vec2 res = u_resolution;
  float T = u_time * 0.3; // Slower base rotation
  
  // Audio-reactive scaling with increased dynamics
  float audioScale = 1.0 + (u_audioLevel * 0.5 + u_bassLevel * 0.4) * u_scaleReactivity * u_intensity;
  
  // Normalize coordinates with audio scaling
  vec2 uv = (FC - 0.5 * res) / res.y * audioScale;
  
  // Ray direction setup
  vec3 rd = normalize(vec3(uv, 1.0));
  
  // Initialize variables closer to original equation
  vec4 O = vec4(0.0); // Start with black
  float i = 0.0;
  float t = 0.0;
  float d = 0.0;
  vec3 p;
  vec3 R;
  
  // Main tornado ray marching loop - closer to original
  // Original: for(O*=i;++i<60.;)p=t*.5*rd,p.z-=3.,p.zx=mod(p.zx*r(T)+9.,18.)-9.,p.zx*=r(round(atan(p.z,p.x)*7.)/7.),R=abs(p),p.x-=pow(R.y*.5,2.5)+sin(p.y*4.+T*8.)*sin(T)*.4,p.y=mod(p.y+T,.2)-.1,t+=d=abs(length(p)-.1)+.01,O+=.02/++d*(1.+cos(R.y+4.+vec4(0,1,2,0)));
  
  O *= i; // Start with original initialization
  
  for(; ++i < 60.0;) {
    // Enhanced audio dynamics on ray position
    p = t * 0.5 * rd;
    p.z -= 3.0 + u_bassLevel * 1.0 * u_intensity; // More bass response
    
    // Rotation with audio-reactive speed (slower base, more audio variation)
    float rotSpeed = T * (0.5 + u_rotationSpeed * 0.5 + u_bassLevel * 0.8 * u_intensity);
    p.zx = mod(p.zx * r(rotSpeed) + 9.0, 18.0) - 9.0;
    
    // Quantization with audio modulation
    float angleQuant = 7.0 + u_midLevel * 3.0 * u_intensity; // More dynamic
    p.zx *= r(round(atan(p.z, p.x) * angleQuant) / angleQuant);
    
    R = abs(p);
    
    // Tornado shape closer to original with enhanced audio response
    float audioMod = u_bassLevel * 0.3 + getFrequency(i / 60.0) * 0.2;
    p.x -= pow(R.y * 0.5, 2.5 + audioMod * u_intensity) + 
           sin(p.y * 4.0 + T * 8.0 * (1.0 + u_trebleLevel * 0.5)) * 
           sin(T * (1.0 + u_midLevel * 0.3)) * (0.4 + audioMod * u_intensity);
    
    // Vertical movement with more audio dynamics
    p.y = mod(p.y + T * (1.0 + u_audioLevel * 0.5 * u_intensity), 0.2) - 0.1;
    
    // Update distance field
    t += d = abs(length(p) - 0.1) + 0.01;
    
    // Color accumulation closer to original but with enhanced audio response
    float beatFactor = 1.0 + u_bassLevel * u_bassLevel * 0.8 * u_beatPulse; // More beat response
    O += 0.02 / ++d * (1.0 + cos(R.y + 4.0 + vec4(0, 1, 2, 0) + u_audioLevel * 2.0 * u_colorSensitivity)) * beatFactor;
  }
  
  // Reduced brightness color output with enhanced audio dynamics
  vec3 color = vec3(0);
  
  // Storm colors with more audio variation but lower brightness
  float hueShift = u_audioLevel * 0.1 * u_colorSensitivity + u_bassLevel * 0.05;
  float saturation = 0.7 + u_midLevel * 0.2 * u_colorSensitivity; // More saturated
  float brightness = 0.35; // Base brightness reduced
  
  // Apply colors with reduced brightness multipliers
  color += hsv(O.x * 0.1 + T * 0.02 + 0.6 + hueShift, saturation, O.x * brightness * (1.0 + u_bassLevel * 0.3 * u_intensity));
  color += hsv(O.y * 0.1 + T * 0.02 + 0.55 + hueShift, saturation * 0.9, O.y * brightness * 0.8 * (1.0 + u_midLevel * 0.25 * u_intensity));
  color += hsv(O.z * 0.1 + T * 0.02 + 0.7 + hueShift, saturation * 0.8, O.z * brightness * 0.6 * (1.0 + u_trebleLevel * 0.2 * u_intensity));
  
  // Darker fog with more audio dynamics
  float fogAmount = smoothstep(0.0, 1.0, t * 0.01);
  vec3 fogColor = vec3(0.01, 0.01, 0.015) + vec3(u_audioLevel * 0.02 * u_intensity);
  color = mix(color, fogColor, fogAmount * (0.2 + u_audioLevel * 0.2 * u_intensity));
  
  // Enhanced lightning with lower brightness
  float lightningChance = getFrequency(0.8) * u_beatPulse + u_trebleLevel * u_beatPulse;
  if (lightningChance > 0.6) { // More frequent but dimmer
    float lightningIntensity = pow(lightningChance, 2.0) * 0.2 * u_intensity; // Reduced brightness
    color += vec3(0.2, 0.25, 0.5) * lightningIntensity;
  }
  
  // More dynamic final audio enhancement
  color *= 0.8 + u_audioLevel * 0.15 * u_intensity + u_bassLevel * 0.1 * u_intensity;
  
  // Waveform overlay with reduced brightness
  vec2 cornerPos = FC / res;
  if (cornerPos.x < 0.12 && cornerPos.y > 0.88) {
    float waveIndex = cornerPos.x * 8.33;
    int waveIdx = int(waveIndex * 31.0);
    float waveValue = u_waveformData[waveIdx];
    color += vec3(0.03, 0.03, 0.06) * abs(waveValue) * 1.5 * u_intensity;
  }
  
  // Final clamp to prevent bright flashes - lower maximum
  color = clamp(color, vec3(0.0), vec3(0.45));
  
  outColor = vec4(color, 1.0);
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

    let cleanup = () => {}

    try {
      vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource)
      fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource)

      if (!vertexShader || !fragmentShader) {
        cleanup = () => {
          window.removeEventListener("resize", resizeCanvas)
        }
        return
      }

      // Create program and link shaders
      program = gl.createProgram()
      if (!program) {
        console.error("Failed to create program")
        cleanup = () => {
          window.removeEventListener("resize", resizeCanvas)
        }
        return
      }

      gl.attachShader(program, vertexShader)
      gl.attachShader(program, fragmentShader)
      gl.linkProgram(program)

      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error("Program linking error:", gl.getProgramInfoLog(program))
        cleanup = () => {
          window.removeEventListener("resize", resizeCanvas)
        }
        return
      }

      // Set up position buffer (full screen quad)
      positionBuffer = gl.createBuffer()
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)

      const positions = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]

      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW)

      // Use program
      gl.useProgram(program)

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
      
      // Control uniform locations
      const intensityUniformLocation = gl.getUniformLocation(program, "u_intensity")
      const rotationSpeedUniformLocation = gl.getUniformLocation(program, "u_rotationSpeed")
      const colorSensitivityUniformLocation = gl.getUniformLocation(program, "u_colorSensitivity")
      const beatPulseUniformLocation = gl.getUniformLocation(program, "u_beatPulse")
      const fractalComplexityUniformLocation = gl.getUniformLocation(program, "u_fractalComplexity")
      const scaleReactivityUniformLocation = gl.getUniformLocation(program, "u_scaleReactivity")

      // Render loop
      const startTime = Date.now()

      const render = () => {
        if (!gl || !program) return

        const currentTime = Date.now()
        const deltaTime = (currentTime - startTime) / 1000

        // Get current audio data
        const audioData = analyzeAudio()

        // Update time uniform
        gl.uniform1f(timeUniformLocation, deltaTime)

        // Update resolution uniform
        gl.uniform2f(resolutionUniformLocation, canvas.width, canvas.height)

        // Update audio uniforms
        if (audioLevelUniformLocation) {
          gl.uniform1f(audioLevelUniformLocation, audioData.level)
        }
        if (bassLevelUniformLocation) {
          gl.uniform1f(bassLevelUniformLocation, audioData.bassLevel)
        }
        if (midLevelUniformLocation) {
          gl.uniform1f(midLevelUniformLocation, audioData.midLevel)
        }
        if (trebleLevelUniformLocation) {
          gl.uniform1f(trebleLevelUniformLocation, audioData.trebleLevel)
        }
        if (frequencyDataUniformLocation) {
          gl.uniform1fv(frequencyDataUniformLocation, audioData.frequencyData)
        }
        if (waveformDataUniformLocation) {
          gl.uniform1fv(waveformDataUniformLocation, audioData.waveformData)
        }

        // Update control uniforms
        if (intensityUniformLocation) {
          gl.uniform1f(intensityUniformLocation, intensity)
        }
        if (rotationSpeedUniformLocation) {
          gl.uniform1f(rotationSpeedUniformLocation, rotationSpeed)
        }
        if (colorSensitivityUniformLocation) {
          gl.uniform1f(colorSensitivityUniformLocation, colorSensitivity)
        }
        if (beatPulseUniformLocation) {
          gl.uniform1f(beatPulseUniformLocation, beatPulse)
        }
        if (fractalComplexityUniformLocation) {
          gl.uniform1f(fractalComplexityUniformLocation, fractalComplexity)
        }
        if (scaleReactivityUniformLocation) {
          gl.uniform1f(scaleReactivityUniformLocation, scaleReactivity)
        }

        // Clear canvas and draw
        gl.clearColor(0, 0, 0, 1)
        gl.clear(gl.COLOR_BUFFER_BIT)
        gl.drawArrays(gl.TRIANGLES, 0, 6)

        requestAnimationFrame(render)
      }

      render()

      // Cleanup
      cleanup = () => {
        window.removeEventListener("resize", resizeCanvas)

        if (program) gl.deleteProgram(program)
        if (vertexShader) gl.deleteShader(vertexShader)
        if (fragmentShader) gl.deleteShader(fragmentShader)
        if (positionBuffer) gl.deleteBuffer(positionBuffer)
      }
    } catch (error) {
      console.error("Error during WebGL initialization or rendering:", error)
      cleanup = () => {
        window.removeEventListener("resize", resizeCanvas)
      }
    }

    return () => {
      cleanup()
    }
  }, [analyzeAudio, intensity, rotationSpeed, colorSensitivity, beatPulse, fractalComplexity, scaleReactivity])

  // Cleanup on unmount
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
      style={{ position: "relative", width: "100vw", height: "100vh" }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <canvas
        ref={canvasRef}
        style={{
          display: "block",
          width: "100%",
          height: "100%",
          position: "absolute",
          top: 0,
          left: 0,
        }}
      />

      {/* Audio Controls */}
      <div
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          background: "rgba(0, 0, 0, 0.8)",
          borderRadius: "10px",
          padding: "15px",
          color: "white",
          fontFamily: "sans-serif",
          fontSize: "14px",
          minWidth: "300px",
          zIndex: 10,
          backdropFilter: "blur(10px)",
          border: "1px solid rgba(255, 255, 255, 0.1)"
        }}
      >
        {/* File Upload */}
        <div style={{ marginBottom: "15px" }}>
          <label style={{ display: "block", marginBottom: "8px", fontSize: "12px", opacity: 0.8 }}>
            Upload Audio File
          </label>
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            style={{
              width: "100%",
              padding: "8px",
              background: "rgba(255, 255, 255, 0.1)",
              border: "1px solid rgba(255, 255, 255, 0.3)",
              borderRadius: "5px",
              color: "white",
              fontSize: "12px"
            }}
          />
        </div>

        {/* Default Songs */}
        <div style={{ marginBottom: "15px" }}>
          <label style={{ display: "block", marginBottom: "8px", fontSize: "12px", opacity: 0.8 }}>
            Or Choose a Default Song
          </label>
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            {defaultSongs.map((song, index) => (
              <button
                key={index}
                onClick={() => loadDefaultSong(song.filename, song.displayName)}
                style={{
                  width: "100%",
                  padding: "8px",
                  background: "rgba(255, 255, 255, 0.1)",
                  border: "1px solid rgba(255, 255, 255, 0.3)",
                  borderRadius: "5px",
                  color: "white",
                  fontSize: "12px",
                  textAlign: "left",
                  cursor: "pointer",
                  transition: "background-color 0.2s"
                }}
                onMouseEnter={(e) => {
                  (e.target as HTMLButtonElement).style.background = "rgba(255, 255, 255, 0.2)"
                }}
                onMouseLeave={(e) => {
                  (e.target as HTMLButtonElement).style.background = "rgba(255, 255, 255, 0.1)"
                }}
              >
                {song.displayName}
              </button>
            ))}
          </div>
        </div>

        {/* Track Info */}
        {trackName && (
          <div style={{ marginBottom: "15px", fontSize: "13px", fontWeight: "bold" }}>
            {trackName}
          </div>
        )}

        {/* Play/Pause Button */}
        <div style={{ marginBottom: "15px", textAlign: "center" }}>
          <button
            onClick={togglePlayPause}
            disabled={!audioElementRef.current}
            style={{
              padding: "10px 20px",
              background: isPlaying ? "#ff4444" : "#44ff44",
              border: "none",
              borderRadius: "20px",
              color: "white",
              fontWeight: "bold",
              cursor: "pointer",
              fontSize: "14px",
              opacity: !audioElementRef.current ? 0.5 : 1
            }}
          >
            {isPlaying ? "⏸ Pause" : "▶ Play"}
          </button>
        </div>

        {/* Timeline */}
        {duration > 0 && (
          <div style={{ marginBottom: "15px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "5px" }}>
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={(currentTime / duration) * 100}
              onChange={handleSeek}
              style={{
                width: "100%",
                height: "4px",
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>
        )}

        {/* Volume Control */}
        <div style={{ marginBottom: "15px" }}>
          <label style={{ display: "block", marginBottom: "5px", fontSize: "12px", opacity: 0.8 }}>
            Volume: {Math.round(volume * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={volume * 100}
            onChange={handleVolumeChange}
            style={{
              width: "100%",
              height: "4px",
              background: "rgba(255, 255, 255, 0.3)",
              outline: "none",
              borderRadius: "2px"
            }}
          />
        </div>

        {/* Visual Controls */}
        <div style={{ marginBottom: "15px", borderTop: "1px solid rgba(255, 255, 255, 0.2)", paddingTop: "15px" }}>
          <div style={{ marginBottom: "8px", fontSize: "13px", fontWeight: "bold", opacity: 0.9 }}>
            Visual Controls
          </div>

          {/* Intensity */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Intensity: {Math.round(intensity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={intensity * 100}
              onChange={(e) => setIntensity(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Rotation Speed */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Rotation Speed: {Math.round(rotationSpeed * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={rotationSpeed * 100}
              onChange={(e) => setRotationSpeed(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Color Sensitivity */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Color Sensitivity: {Math.round(colorSensitivity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={colorSensitivity * 100}
              onChange={(e) => setColorSensitivity(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Beat Pulse */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Beat Pulse: {Math.round(beatPulse * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={beatPulse * 100}
              onChange={(e) => setBeatPulse(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Fractal Complexity */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Fractal Complexity: {Math.round(fractalComplexity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="150"
              value={fractalComplexity * 100}
              onChange={(e) => setFractalComplexity(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Scale Reactivity */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Scale Reactivity: {Math.round(scaleReactivity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={scaleReactivity * 100}
              onChange={(e) => setScaleReactivity(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Smoothing */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Smoothing: {Math.round(smoothing * 100)}%
            </label>
            <input
              type="range"
              min="10"
              max="95"
              value={smoothing * 100}
              onChange={(e) => setSmoothing(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Reset Button */}
          <div style={{ textAlign: "center", marginTop: "10px" }}>
            <button
              onClick={() => {
                setIntensity(0.5)
                setRotationSpeed(0.5)
                setColorSensitivity(0.5)
                setBeatPulse(0.5)
                setFractalComplexity(0.5)
                setSmoothing(0.85)
                setScaleReactivity(0.5)
              }}
              style={{
                padding: "6px 12px",
                background: "rgba(255, 255, 255, 0.2)",
                border: "1px solid rgba(255, 255, 255, 0.3)",
                borderRadius: "4px",
                color: "white",
                fontSize: "11px",
                cursor: "pointer"
              }}
            >
              Reset Controls
            </button>
          </div>
        </div>

        {/* Audio Levels Display */}
        {isPlaying && (
          <div style={{ fontSize: "10px", opacity: 0.7 }}>
            <div>Level: {Math.round(smoothedAudioDataRef.current.level * 100)}%</div>
            <div>Bass: {Math.round(smoothedAudioDataRef.current.bassLevel * 100)}%</div>
            <div>Mid: {Math.round(smoothedAudioDataRef.current.midLevel * 100)}%</div>
            <div>Treble: {Math.round(smoothedAudioDataRef.current.trebleLevel * 100)}%</div>
          </div>
        )}
      </div>

      {/* Drag and Drop Overlay */}
      {isDragOver && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            background: "rgba(0, 100, 255, 0.2)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 20,
            fontSize: "24px",
            color: "white",
            fontFamily: "sans-serif",
            fontWeight: "bold"
          }}
        >
          Drop audio file here
        </div>
      )}

      {/* Attribution */}
      <div style={{ position: 'absolute', bottom: '10px', right: '10px', color: 'white', fontSize: '12px', opacity: 0.7 }}>
        <div>Tornado GLSL Visualization</div>
        <div style={{ fontSize: '10px', marginTop: '2px' }}>
          Audio-Reactive Enhanced
        </div>
      </div>
    </div>
  )
};

export default TornadoGLSLVisualization; 