"use client"

import { useRef, useEffect, useState, useCallback } from "react"

interface AudioData {
  level: number
  bassLevel: number
  midLevel: number
  trebleLevel: number
  frequencyData: Float32Array
  waveformData: Float32Array
}

export default function KaleidoscopeGLSLVisualization() {
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
  const audioLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const bassLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const midLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const trebleLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const frequencyDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const waveformDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)

  // Control uniform locations
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

    const bassEnd = Math.floor(bufferLength * 0.1)
    const midEnd = Math.floor(bufferLength * 0.5)

    let bassSum = 0, midSum = 0, trebleSum = 0, totalSum = 0

    for (let i = 0; i < bassEnd; i++) {
      bassSum += dataArray[i]
    }
    bassSum /= bassEnd

    for (let i = bassEnd; i < midEnd; i++) {
      midSum += dataArray[i]
    }
    midSum /= (midEnd - bassEnd)

    for (let i = midEnd; i < bufferLength; i++) {
      trebleSum += dataArray[i]
    }
    trebleSum /= (bufferLength - midEnd)

    for (let i = 0; i < bufferLength; i++) {
      totalSum += dataArray[i]
    }
    const level = totalSum / bufferLength / 255

    const bassLevel = bassSum / 255
    const midLevel = midSum / 255
    const trebleLevel = trebleSum / 255

    for (let i = 0; i < 64; i++) {
      const index = Math.floor((i / 64) * bufferLength)
      frequencyDataRef.current[i] = dataArray[index] / 255
    }

    for (let i = 0; i < 32; i++) {
      const index = Math.floor((i / 32) * bufferLength)
      waveformDataRef.current[i] = (waveformArray[index] - 128) / 128
    }

    const smoothingFactor = smoothing
    const current = smoothedAudioDataRef.current

    current.level = current.level * smoothingFactor + level * (1 - smoothingFactor)
    current.bassLevel = current.bassLevel * smoothingFactor + bassLevel * (1 - smoothingFactor)
    current.midLevel = current.midLevel * smoothingFactor + midLevel * (1 - smoothingFactor)
    current.trebleLevel = current.trebleLevel * smoothingFactor + trebleLevel * (1 - smoothingFactor)

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

    if (audioElementRef.current) {
      audioElementRef.current.pause()
      if (audioElementRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioElementRef.current.src)
      }
    }

    audioElementRef.current = audio

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

    if (audioElementRef.current) {
      audioElementRef.current.pause()
      if (audioElementRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioElementRef.current.src)
      }
    }

    audioElementRef.current = audio

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

  // WebGL setup
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const gl = canvas.getContext("webgl2")
    if (!gl) {
      console.error("WebGL2 not supported")
      return
    }

    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
      gl.viewport(0, 0, canvas.width, canvas.height)
    }

    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    const vertexShaderSource = `#version 300 es
      precision highp float;
      in vec4 a_position;
      void main() {
        gl_Position = a_position;
      }
    `

    // Enhanced Kaleidoscope fragment shader
    const fragmentShaderSource = `#version 300 es
      precision highp float;
      out vec4 outColor;
      
      uniform vec2 u_resolution;
      uniform float u_time;
      uniform float u_audioLevel;
      uniform float u_bassLevel;
      uniform float u_midLevel;
      uniform float u_trebleLevel;
      uniform float u_frequencyData[64];
      uniform float u_waveformData[32];
      uniform float u_intensity;
      uniform float u_rotationSpeed;
      uniform float u_colorSensitivity;
      uniform float u_beatPulse;
      uniform float u_fractalComplexity;
      uniform float u_scaleReactivity;
      
      #define PI 3.14159265359
      #define TAU 6.28318530718
      
      // HSV to RGB conversion
      vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
      }
      
      // Get frequency for normalized position
      float getFreq(float pos) {
        int idx = int(clamp(pos * 63.0, 0.0, 63.0));
        return u_frequencyData[idx];
      }
      
      // Kaleidoscope symmetry function
      vec2 kaleidoscope(vec2 uv, float segments) {
        float angle = atan(uv.y, uv.x);
        float radius = length(uv);
        
        // Create kaleidoscope segments
        float segmentAngle = TAU / segments;
        angle = mod(angle, segmentAngle);
        
        // Mirror every other segment
        if (mod(floor(atan(uv.y, uv.x) / segmentAngle), 2.0) > 0.5) {
          angle = segmentAngle - angle;
        }
        
        return vec2(cos(angle), sin(angle)) * radius;
      }
      
      // Complex pattern generator
      float pattern(vec2 uv, float time, float audioReactivity) {
        // Multi-layered geometric patterns
        float pattern1 = sin(uv.x * 20.0 + time * 2.0) * cos(uv.y * 15.0 + time * 1.5);
        float pattern2 = sin(length(uv) * 30.0 - time * 3.0) * 0.5;
        float pattern3 = sin((uv.x + uv.y) * 25.0 + time * 2.0) * cos((uv.x - uv.y) * 18.0 - time * 1.8);
        
        // Audio-reactive modulation
        pattern1 *= 1.0 + audioReactivity * 0.3;
        pattern2 *= 1.0 + audioReactivity * 0.5;
        pattern3 *= 1.0 + audioReactivity * 0.4;
        
        return (pattern1 + pattern2 + pattern3) / 3.0;
      }
      
      // Sharp line generator
      float sharpLine(vec2 uv, vec2 start, vec2 end, float thickness) {
        vec2 pa = uv - start;
        vec2 ba = end - start;
        float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        float dist = length(pa - ba * h);
        return 1.0 - smoothstep(0.0, thickness, dist);
      }
      
      void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);
        
        // Audio-reactive scaling and rotation
        float audioScale = 1.0 + (u_audioLevel * 0.3 + u_bassLevel * 0.4) * u_scaleReactivity;
        uv *= audioScale;
        
        // Main rotation based on time and audio
        float rotTime = u_time * u_rotationSpeed * (0.5 + u_midLevel * 0.5);
        mat2 rot = mat2(cos(rotTime), sin(rotTime), -sin(rotTime), cos(rotTime));
        uv = rot * uv;
        
        // Dynamic kaleidoscope segments based on audio
        float segments = 6.0 + floor(u_bassLevel * 8.0 * u_fractalComplexity);
        vec2 kUv = kaleidoscope(uv, segments);
        
        // Multiple pattern layers
        float mainPattern = pattern(kUv * 2.0, u_time, u_audioLevel * u_intensity);
        float detailPattern = pattern(kUv * 5.0, u_time * 1.3, u_trebleLevel * u_intensity);
        float finePattern = pattern(kUv * 12.0, u_time * 0.8, u_midLevel * u_intensity);
        
        // Sharp geometric lines
        float lines = 0.0;
        for (float i = 0.0; i < 8.0; i++) {
          float angle = i * TAU / 8.0 + u_time * u_rotationSpeed * 0.5;
          vec2 dir = vec2(cos(angle), sin(angle));
          float lineIntensity = getFreq(i / 8.0) * u_intensity;
          lines += sharpLine(kUv, vec2(0.0), dir * (0.5 + lineIntensity * 0.3), 0.02 + lineIntensity * 0.01);
        }
        
        // Radial lines
        float radialLines = 0.0;
        float radius = length(kUv);
        float angle = atan(kUv.y, kUv.x);
        for (float i = 0.0; i < 16.0; i++) {
          float lineAngle = i * TAU / 16.0;
          float angleDiff = abs(angle - lineAngle);
          angleDiff = min(angleDiff, TAU - angleDiff);
          float freqValue = getFreq(i / 16.0);
          radialLines += (1.0 - smoothstep(0.0, 0.1 - freqValue * 0.05, angleDiff)) * freqValue;
        }
        
        // Concentric circles
        float circles = 0.0;
        for (float i = 1.0; i <= 8.0; i++) {
          float circleRadius = i * 0.15;
          float freqValue = getFreq(i / 8.0);
          float circleWidth = 0.02 + freqValue * 0.01 * u_intensity;
          circles += (1.0 - smoothstep(0.0, circleWidth, abs(radius - circleRadius))) * freqValue;
        }
        
        // Combine all patterns
        float combinedPattern = mainPattern * 0.4 + detailPattern * 0.3 + finePattern * 0.3;
        combinedPattern += lines * 0.6 + radialLines * 0.4 + circles * 0.5;
        
        // Audio-reactive color mapping
        float hue1 = 0.1 + u_bassLevel * 0.3 * u_colorSensitivity + u_time * 0.1;
        float hue2 = 0.5 + u_midLevel * 0.4 * u_colorSensitivity + u_time * 0.05;
        float hue3 = 0.8 + u_trebleLevel * 0.2 * u_colorSensitivity - u_time * 0.03;
        
        // Multi-hue blending
        vec3 color1 = hsv2rgb(vec3(hue1, 0.8 + u_audioLevel * 0.2, combinedPattern * 2.0));
        vec3 color2 = hsv2rgb(vec3(hue2, 0.9, combinedPattern * 1.5));
        vec3 color3 = hsv2rgb(vec3(hue3, 0.7 + u_trebleLevel * 0.3, combinedPattern * 1.8));
        
        // Blend colors based on frequency content
        vec3 finalColor = mix(mix(color1, color2, u_midLevel * u_colorSensitivity), color3, u_trebleLevel * u_colorSensitivity);
        
        // Beat pulse enhancement
        float beatPulse = 1.0 + u_bassLevel * u_bassLevel * 0.8 * u_beatPulse;
        finalColor *= beatPulse;
        
        // Sharp contrast enhancement
        finalColor = pow(finalColor, vec3(0.8 + u_audioLevel * 0.3));
        
        // Final intensity boost
        finalColor *= 1.0 + u_audioLevel * 0.2 * u_intensity;
        
        outColor = vec4(finalColor, 1.0);
      }
    `

    // Create and compile shaders
    const createShader = (gl: WebGL2RenderingContext, type: number, source: string) => {
      const shader = gl.createShader(type)
      if (!shader) return null
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

    const program = gl.createProgram()
    if (!program) return

    gl.attachShader(program, vertexShader)
    gl.attachShader(program, fragmentShader)
    gl.linkProgram(program)

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("Program linking error:", gl.getProgramInfoLog(program))
      return
    }

    programRef.current = program

    // Set up vertex buffer
    const positionBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
    const positions = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW)

    const positionAttributeLocation = gl.getAttribLocation(program, "a_position")
    gl.enableVertexAttribArray(positionAttributeLocation)
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0)

    // Get uniform locations
    resolutionUniformLocationRef.current = gl.getUniformLocation(program, "u_resolution")
    timeUniformLocationRef.current = gl.getUniformLocation(program, "u_time")
    audioLevelUniformLocationRef.current = gl.getUniformLocation(program, "u_audioLevel")
    bassLevelUniformLocationRef.current = gl.getUniformLocation(program, "u_bassLevel")
    midLevelUniformLocationRef.current = gl.getUniformLocation(program, "u_midLevel")
    trebleLevelUniformLocationRef.current = gl.getUniformLocation(program, "u_trebleLevel")
    frequencyDataUniformLocationRef.current = gl.getUniformLocation(program, "u_frequencyData")
    waveformDataUniformLocationRef.current = gl.getUniformLocation(program, "u_waveformData")
    intensityUniformLocationRef.current = gl.getUniformLocation(program, "u_intensity")
    rotationSpeedUniformLocationRef.current = gl.getUniformLocation(program, "u_rotationSpeed")
    colorSensitivityUniformLocationRef.current = gl.getUniformLocation(program, "u_colorSensitivity")
    beatPulseUniformLocationRef.current = gl.getUniformLocation(program, "u_beatPulse")
    fractalComplexityUniformLocationRef.current = gl.getUniformLocation(program, "u_fractalComplexity")
    scaleReactivityUniformLocationRef.current = gl.getUniformLocation(program, "u_scaleReactivity")

    const startTime = performance.now()

    const render = () => {
      if (!programRef.current || !gl) return
      gl.useProgram(programRef.current)

      const audioData = analyzeAudio()
      const currentTime = (performance.now() - startTime) / 1000

      // Update all uniforms
      if (timeUniformLocationRef.current) gl.uniform1f(timeUniformLocationRef.current, currentTime)
      if (resolutionUniformLocationRef.current) gl.uniform2f(resolutionUniformLocationRef.current, canvas.width, canvas.height)
      if (audioLevelUniformLocationRef.current) gl.uniform1f(audioLevelUniformLocationRef.current, audioData.level)
      if (bassLevelUniformLocationRef.current) gl.uniform1f(bassLevelUniformLocationRef.current, audioData.bassLevel)
      if (midLevelUniformLocationRef.current) gl.uniform1f(midLevelUniformLocationRef.current, audioData.midLevel)
      if (trebleLevelUniformLocationRef.current) gl.uniform1f(trebleLevelUniformLocationRef.current, audioData.trebleLevel)
      if (frequencyDataUniformLocationRef.current) gl.uniform1fv(frequencyDataUniformLocationRef.current, audioData.frequencyData)
      if (waveformDataUniformLocationRef.current) gl.uniform1fv(waveformDataUniformLocationRef.current, audioData.waveformData)
      if (intensityUniformLocationRef.current) gl.uniform1f(intensityUniformLocationRef.current, intensity)
      if (rotationSpeedUniformLocationRef.current) gl.uniform1f(rotationSpeedUniformLocationRef.current, rotationSpeed)
      if (colorSensitivityUniformLocationRef.current) gl.uniform1f(colorSensitivityUniformLocationRef.current, colorSensitivity)
      if (beatPulseUniformLocationRef.current) gl.uniform1f(beatPulseUniformLocationRef.current, beatPulse)
      if (fractalComplexityUniformLocationRef.current) gl.uniform1f(fractalComplexityUniformLocationRef.current, fractalComplexity)
      if (scaleReactivityUniformLocationRef.current) gl.uniform1f(scaleReactivityUniformLocationRef.current, scaleReactivity)

      gl.clearColor(0, 0, 0, 1)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.drawArrays(gl.TRIANGLES, 0, 6)

      animationRef.current = requestAnimationFrame(render)
    }

    render()

    return () => {
      window.removeEventListener("resize", resizeCanvas)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      if (program) {
        gl.deleteProgram(program)
      }
      gl.deleteShader(vertexShader)
      gl.deleteShader(fragmentShader)
      gl.deleteBuffer(positionBuffer)
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

      {/* Audio Controls - Same as Spaceship but with Kaleidoscope branding */}
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
        {/* Component Title */}
        <div style={{ marginBottom: "15px", fontSize: "16px", fontWeight: "bold", textAlign: "center" }}>
          🔮 Kaleidoscope Visualizer
        </div>

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
            Kaleidoscope Controls
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

          {/* Fractal Complexity (now controls kaleidoscope segments) */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Kaleidoscope Segments: {Math.round(fractalComplexity * 100)}%
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
    </div>
  )
} 