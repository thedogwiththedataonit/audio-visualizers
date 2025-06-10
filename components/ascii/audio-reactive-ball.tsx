"use client"

import type React from "react"

import { useEffect, useRef, useState, useCallback } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Settings, Play, Pause, Music, Mic, Volume2, VolumeX } from "lucide-react"

export default function Component() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const asciiRef = useRef<HTMLDivElement>(null)
  const animationRef = useRef<number | undefined>(undefined)

  // Audio state and refs
  const audioContextRef = useRef<AudioContext | undefined>(undefined)
  const analyserRef = useRef<AnalyserNode | undefined>(undefined)
  const dataArrayRef = useRef<Uint8Array | undefined>(undefined)
  const micSourceRef = useRef<MediaStreamAudioSourceNode | undefined>(undefined)
  const micGainNodeRef = useRef<GainNode | undefined>(undefined)
  const micStreamRef = useRef<MediaStream | undefined>(undefined)
  const audioElementRef = useRef<HTMLAudioElement | null>(null)
  const mp3SourceNodeRef = useRef<MediaElementAudioSourceNode | null>(null)

  // UI state
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [currentAudioSourceType, setCurrentAudioSourceType] = useState<"mic" | "mp3">("mic")
  const [currentPlayingFile, setCurrentPlayingFile] = useState<File | null>(null)
  const [currentPlayingSong, setCurrentPlayingSong] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playerVolume, setPlayerVolume] = useState(1)

  const isSwitchingAudioRef = useRef(false)

  const asciiChars = " .:-=+*#%@"
  const ballRef = useRef({
    x: 0,
    y: 0,
    baseRadius: 100,
    currentRadius: 100,
    targetRadius: 100,
    hue: 200,
    targetHue: 200,
    particles: [] as Array<{
      x: number
      y: number
      vx: number
      vy: number
      life: number
      maxLife: number
      size: number
    }>,
  })

  // Default songs available in the app
  const defaultSongs = [
    { filename: "Charli XCX - party 4 u.mp3", displayName: "Charli XCX - Party 4 U" },
    { filename: "Benson Boone - Beautiful Things.mp3", displayName: "Benson Boone - Beautiful Things" },
    { filename: "M83 - Midnight City.mp3", displayName: "M83 - Midnight City" }
  ]

  const formatTime = (timeInSeconds: number) => {
    if (isNaN(timeInSeconds) || timeInSeconds === Number.POSITIVE_INFINITY) return "0:00"
    const minutes = Math.floor(timeInSeconds / 60)
    const seconds = Math.floor(timeInSeconds % 60)
    return `${minutes}:${seconds.toString().padStart(2, "0")}`
  }

  const initAudioContextAndAnalyser = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 48000,
      })
    }
    if (audioContextRef.current.state === "suspended") {
      await audioContextRef.current.resume()
    }
    if (!analyserRef.current && audioContextRef.current) {
      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 256
      analyserRef.current.smoothingTimeConstant = 0.5
      dataArrayRef.current = new Uint8Array(analyserRef.current.frequencyBinCount)
    }
  }, [])

  const disconnectMic = useCallback(() => {
    if (micSourceRef.current && micGainNodeRef.current && analyserRef.current) {
      try {
        micSourceRef.current.disconnect(micGainNodeRef.current)
      } catch (e) {}
      try {
        micGainNodeRef.current.disconnect(analyserRef.current)
      } catch (e) {}
    } else if (micSourceRef.current && analyserRef.current) {
      try {
        micSourceRef.current.disconnect(analyserRef.current)
      } catch (e) {}
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((track) => track.stop())
      micStreamRef.current = undefined
    }
  }, [])

  const disconnectMp3 = useCallback(() => {
    if (mp3SourceNodeRef.current && analyserRef.current && audioContextRef.current) {
      try {
        mp3SourceNodeRef.current.disconnect(analyserRef.current)
      } catch (e) {}
      try {
        mp3SourceNodeRef.current.disconnect(audioContextRef.current.destination)
      } catch (e) {}
    }
    if (audioElementRef.current) {
      audioElementRef.current.pause()
    }
  }, [])

  const initMicAudio = useCallback(async () => {
    await initAudioContextAndAnalyser()
    if (!audioContextRef.current || !analyserRef.current) return

    disconnectMp3()

    try {
      const constraints = {
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          sampleRate: 48000,
          channelCount: 1,
          volume: 1.0,
        },
      }
      if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach((track) => track.stop())
      }
      micStreamRef.current = await navigator.mediaDevices.getUserMedia(constraints)

      if (!micSourceRef.current || micSourceRef.current.mediaStream !== micStreamRef.current) {
        if (micSourceRef.current && micGainNodeRef.current) {
          try {
            micSourceRef.current.disconnect(micGainNodeRef.current)
          } catch (e) {}
        } else if (micSourceRef.current) {
          try {
            micSourceRef.current.disconnect()
          } catch (e) {}
        }
        micSourceRef.current = audioContextRef.current.createMediaStreamSource(micStreamRef.current)
      }

      if (!micGainNodeRef.current) {
        micGainNodeRef.current = audioContextRef.current.createGain()
        micGainNodeRef.current.gain.value = 3.0
      }

      micSourceRef.current.connect(micGainNodeRef.current)
      micGainNodeRef.current.connect(analyserRef.current)

      setCurrentAudioSourceType("mic")
      setCurrentPlayingFile(null)
      setCurrentPlayingSong(null)
      setIsPlaying(false)
    } catch (err) {
      console.error("Error accessing microphone:", err)
    }
  }, [initAudioContextAndAnalyser, disconnectMp3])

  const setupMp3Audio = useCallback(
    async (file: File) => {
      if (isSwitchingAudioRef.current) {
        return
      }
      isSwitchingAudioRef.current = true

      await initAudioContextAndAnalyser()
      if (!audioContextRef.current || !analyserRef.current) {
        isSwitchingAudioRef.current = false
        return
      }

      disconnectMic()

      if (audioElementRef.current) {
        audioElementRef.current.pause()
      }
      if (mp3SourceNodeRef.current) {
        try {
          mp3SourceNodeRef.current.disconnect()
        } catch (e) {}
      }

      if (!audioElementRef.current) {
        audioElementRef.current = new Audio()
        audioElementRef.current.crossOrigin = "anonymous"
        audioElementRef.current.addEventListener("loadedmetadata", () => {
          if (audioElementRef.current) setDuration(audioElementRef.current.duration || 0)
        })
        audioElementRef.current.addEventListener("timeupdate", () => {
          if (audioElementRef.current) setCurrentTime(audioElementRef.current.currentTime || 0)
        })
        audioElementRef.current.addEventListener("play", () => setIsPlaying(true))
        audioElementRef.current.addEventListener("pause", () => setIsPlaying(false))
        audioElementRef.current.addEventListener("ended", () => setIsPlaying(false))
      }

      if (audioElementRef.current.src && audioElementRef.current.src.startsWith("blob:")) {
        URL.revokeObjectURL(audioElementRef.current.src)
      }
      audioElementRef.current.src = URL.createObjectURL(file)
      audioElementRef.current.volume = playerVolume

      if (!mp3SourceNodeRef.current || mp3SourceNodeRef.current.mediaElement !== audioElementRef.current) {
        mp3SourceNodeRef.current = audioContextRef.current.createMediaElementSource(audioElementRef.current)
      }

      try {
        mp3SourceNodeRef.current.disconnect()
      } catch (e) {}
      mp3SourceNodeRef.current.connect(analyserRef.current)
      mp3SourceNodeRef.current.connect(audioContextRef.current.destination)

      setCurrentAudioSourceType("mp3")
      setCurrentPlayingFile(file)
      setCurrentPlayingSong(null)

      try {
        if (audioContextRef.current && audioContextRef.current.state === "suspended") {
          await audioContextRef.current.resume()
        }
        await audioElementRef.current.play()
      } catch (error: any) {
        if (error.name === "AbortError") {
          console.warn(`Play request for ${file.name} was aborted.`)
        } else {
          console.error(`Error playing audio ${file.name}:`, error)
        }
      } finally {
        isSwitchingAudioRef.current = false
      }
    },
    [initAudioContextAndAnalyser, disconnectMic, playerVolume],
  )

  // Load audio file from URL (for default songs)
  const loadAudioFromUrl = useCallback(async (url: string, displayName: string) => {
    if (isSwitchingAudioRef.current) {
      return
    }
    isSwitchingAudioRef.current = true

    await initAudioContextAndAnalyser()
    if (!audioContextRef.current || !analyserRef.current) {
      isSwitchingAudioRef.current = false
      return
    }

    disconnectMic()

    if (audioElementRef.current) {
      audioElementRef.current.pause()
    }
    if (mp3SourceNodeRef.current) {
      try {
        mp3SourceNodeRef.current.disconnect()
      } catch (e) {}
    }

    if (!audioElementRef.current) {
      audioElementRef.current = new Audio()
      audioElementRef.current.crossOrigin = "anonymous"
      audioElementRef.current.addEventListener("loadedmetadata", () => {
        if (audioElementRef.current) setDuration(audioElementRef.current.duration || 0)
      })
      audioElementRef.current.addEventListener("timeupdate", () => {
        if (audioElementRef.current) setCurrentTime(audioElementRef.current.currentTime || 0)
      })
      audioElementRef.current.addEventListener("play", () => setIsPlaying(true))
      audioElementRef.current.addEventListener("pause", () => setIsPlaying(false))
      audioElementRef.current.addEventListener("ended", () => setIsPlaying(false))
    }

    if (audioElementRef.current.src && audioElementRef.current.src.startsWith("blob:")) {
      URL.revokeObjectURL(audioElementRef.current.src)
    }
    audioElementRef.current.src = url
    audioElementRef.current.volume = playerVolume

    if (!mp3SourceNodeRef.current || mp3SourceNodeRef.current.mediaElement !== audioElementRef.current) {
      mp3SourceNodeRef.current = audioContextRef.current.createMediaElementSource(audioElementRef.current)
    }

    try {
      mp3SourceNodeRef.current.disconnect()
    } catch (e) {}
    mp3SourceNodeRef.current.connect(analyserRef.current)
    mp3SourceNodeRef.current.connect(audioContextRef.current.destination)

    setCurrentAudioSourceType("mp3")
    setCurrentPlayingFile(null)
    setCurrentPlayingSong(displayName)

    try {
      if (audioContextRef.current && audioContextRef.current.state === "suspended") {
        await audioContextRef.current.resume()
      }
      await audioElementRef.current.play()
    } catch (error: any) {
      if (error.name === "AbortError") {
        console.warn(`Play request for ${displayName} was aborted.`)
      } else {
        console.error(`Error playing audio ${displayName}:`, error)
      }
    } finally {
      isSwitchingAudioRef.current = false
    }
  }, [initAudioContextAndAnalyser, disconnectMic, playerVolume])

  // Load default song
  const loadDefaultSong = useCallback((filename: string, displayName: string) => {
    const url = `/songs/${encodeURIComponent(filename)}`
    loadAudioFromUrl(url, displayName)
  }, [loadAudioFromUrl])

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setUploadedFiles((prev) => {
        const newFiles = [...prev]
        if (!newFiles.find((f) => f.name === file.name)) {
          newFiles.push(file)
        }
        return newFiles
      })
      setupMp3Audio(file)
      setIsSettingsOpen(false)
    }
  }

  const handleAudioSourceSelect = (value: string) => {
    if (value === "mic") {
      initMicAudio()
    } else {
      // Check if it's a default song
      const defaultSong = defaultSongs.find((song) => song.displayName === value)
      if (defaultSong) {
        loadDefaultSong(defaultSong.filename, defaultSong.displayName)
      } else {
        // It's an uploaded file
        const selectedFile = uploadedFiles.find((f) => f.name === value)
        if (selectedFile) {
          setupMp3Audio(selectedFile)
        }
      }
    }
    setIsSettingsOpen(false)
  }

  const togglePlayPause = async () => {
    if (isSwitchingAudioRef.current) return
    if (!audioElementRef.current || currentAudioSourceType !== "mp3") return
    if (audioContextRef.current && audioContextRef.current.state === "suspended") {
      await audioContextRef.current.resume()
    }
    if (isPlaying) {
      audioElementRef.current.pause()
    } else {
      try {
        await audioElementRef.current.play()
      } catch (error) {
        console.error("Error playing audio:", error)
      }
    }
  }

  const handleSeek = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioElementRef.current || currentAudioSourceType !== "mp3") return
    const time = Number.parseFloat(event.target.value)
    audioElementRef.current.currentTime = time
    setCurrentTime(time)
  }

  const handleVolumeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioElementRef.current || currentAudioSourceType !== "mp3") return
    const newVolume = Number.parseFloat(event.target.value)
    audioElementRef.current.volume = newVolume
    setPlayerVolume(newVolume)
  }

  const addParticles = useCallback((intensity: number) => {
    const ball = ballRef.current
    const canvas = canvasRef.current
    if (!canvas) return
    const particleCount = Math.floor(intensity * 8)

    for (let i = 0; i < particleCount; i++) {
      const angle = Math.random() * Math.PI * 2
      const speed = 2 + Math.random() * 4
      ball.particles.push({
        x: ball.x + Math.cos(angle) * ball.currentRadius,
        y: ball.y + Math.sin(angle) * ball.currentRadius,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 60,
        maxLife: 60,
        size: 2 + Math.random() * 3,
      })
    }

    if (ball.particles.length > 200) {
      ball.particles.splice(0, ball.particles.length - 200)
    }
  }, [])

  const convertToAscii = useCallback(() => {
    const canvas = canvasRef.current
    const asciiDiv = asciiRef.current
    if (!canvas || !asciiDiv) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    if (canvas.width <= 0 || canvas.height <= 0) return
    const charWidth = 3
    const charHeight = 6
    const cols = Math.floor(canvas.width / charWidth)
    const rows = Math.floor(canvas.height / charHeight)
    if (cols <= 0 || rows <= 0) return
    let imageData
    try {
      imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    } catch (error) {
      console.warn("Failed to get image data:", error)
      return
    }
    const pixels = imageData.data
    let asciiString = ""
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const pixelX = Math.floor(x * charWidth + charWidth / 2)
        const pixelY = Math.floor(y * charHeight + charHeight / 2)
        if (pixelX >= canvas.width || pixelY >= canvas.height) {
          asciiString += " "
          continue
        }
        const pixelIndex = (pixelY * canvas.width + pixelX) * 4
        if (pixelIndex >= pixels.length) {
          asciiString += " "
          continue
        }
        const r = pixels[pixelIndex] || 0
        const g = pixels[pixelIndex + 1] || 0
        const b = pixels[pixelIndex + 2] || 0
        const brightness = (r + g + b) / 3
        const charIndex = Math.floor((brightness / 255) * (asciiChars.length - 1))
        asciiString += asciiChars[charIndex]
      }
      asciiString += "\n"
    }
    asciiDiv.textContent = asciiString
  }, [asciiChars])

  const animate = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      animationRef.current = requestAnimationFrame(animate)
      return
    }
    const ctx = canvas.getContext("2d")
    if (!ctx) {
      animationRef.current = requestAnimationFrame(animate)
      return
    }
    if (canvas.width <= 0 || canvas.height <= 0) {
      animationRef.current = requestAnimationFrame(animate)
      return
    }

    const ball = ballRef.current
    ctx.fillStyle = "rgba(0, 0, 0, 1)"
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    let audioVolume = 0
    let dominantFreq = 0

    if (
      analyserRef.current &&
      dataArrayRef.current &&
      audioContextRef.current &&
      audioContextRef.current.state === "running"
    ) {
      analyserRef.current.getByteFrequencyData(dataArrayRef.current)
      const sum = dataArrayRef.current.reduce((a, b) => a + b, 0)
      audioVolume = sum / dataArrayRef.current.length / 255
      audioVolume = Math.min(1, audioVolume * 3)

      let maxAmplitude = 0
      let maxIndex = 0
      for (let i = 0; i < dataArrayRef.current.length; i++) {
        if (dataArrayRef.current[i] > maxAmplitude) {
          maxAmplitude = dataArrayRef.current[i]
          maxIndex = i
        }
      }
      dominantFreq = maxIndex / dataArrayRef.current.length

      ball.targetRadius = ball.baseRadius + audioVolume * 180
      ball.targetHue = 200 + dominantFreq * 220
      if (audioVolume > 0.05) addParticles(audioVolume)
    } else {
      const time = Date.now() / 1000
      const pulseFactor = Math.sin(time) * 0.1 + 0.9
      ball.targetRadius = ball.baseRadius * pulseFactor
    }

    ball.currentRadius += (ball.targetRadius - ball.currentRadius) * 0.1
    ball.hue += (ball.targetHue - ball.hue) * 0.1
    ball.x = canvas.width / 2
    ball.y = canvas.height / 2

    const gradient = ctx.createRadialGradient(ball.x, ball.y, 0, ball.x, ball.y, ball.currentRadius)
    gradient.addColorStop(0, `rgba(255, 255, 255, 1)`)
    gradient.addColorStop(0.7, `rgba(180, 180, 180, 0.8)`)
    gradient.addColorStop(1, `rgba(80, 80, 80, 0.2)`)
    ctx.shadowColor = `rgba(255, 255, 255, 0.8)`
    ctx.shadowBlur = 30
    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.arc(ball.x, ball.y, ball.currentRadius, 0, Math.PI * 2)
    ctx.fill()
    ctx.shadowBlur = 0
    const coreGradient = ctx.createRadialGradient(ball.x, ball.y, 0, ball.x, ball.y, ball.currentRadius * 0.3)
    coreGradient.addColorStop(0, `rgba(255, 255, 255, 1)`)
    coreGradient.addColorStop(1, `rgba(255, 255, 255, 0.3)`)
    ctx.fillStyle = coreGradient
    ctx.beginPath()
    ctx.arc(ball.x, ball.y, ball.currentRadius * 0.3, 0, Math.PI * 2)
    ctx.fill()

    // Draw frequency bars INSIDE the ball
    if (
      analyserRef.current &&
      dataArrayRef.current &&
      audioContextRef.current &&
      audioContextRef.current.state === "running"
    ) {
      const barCount = 32
      const angleStep = (Math.PI * 2) / barCount
      for (let i = 0; i < barCount; i++) {
        const angle = i * angleStep
        let amplitude = dataArrayRef.current[i * 4] / 255
        amplitude = Math.min(1, amplitude * 3) // Boosted amplitude

        const barInnerLength = amplitude * ball.currentRadius * 0.9

        const startX = ball.x
        const startY = ball.y

        const endX = ball.x + Math.cos(angle) * barInnerLength
        const endY = ball.y + Math.sin(angle) * barInnerLength

        const grayValue = Math.floor(150 + 105 * amplitude) // Adjusted for visibility
        ctx.strokeStyle = `rgba(${grayValue}, ${grayValue}, ${grayValue}, ${amplitude * 0.8})`
        ctx.lineWidth = 2 // Thinner lines
        ctx.shadowColor = `rgba(255, 255, 255, ${amplitude * 0.5})`
        ctx.shadowBlur = 5 // Softer shadow
        ctx.beginPath()
        ctx.moveTo(startX, startY)
        ctx.lineTo(endX, endY)
        ctx.stroke()
      }
    }
    ctx.shadowBlur = 0 // Reset shadowBlur after drawing bars

    // Draw particles (drawn after bars so they can appear on top if desired, or adjust order)
    ball.particles.forEach((particle, index) => {
      particle.x += particle.vx
      particle.y += particle.vy
      particle.life--
      const alpha = particle.life / particle.maxLife
      ctx.fillStyle = `rgba(220, 220, 220, ${alpha})`
      ctx.shadowColor = `rgba(255, 255, 255, ${alpha})`
      ctx.shadowBlur = 8
      ctx.beginPath()
      ctx.arc(particle.x, particle.y, particle.size * alpha, 0, Math.PI * 2)
      ctx.fill()
      if (particle.life <= 0) {
        ball.particles.splice(index, 1)
      }
    })

    convertToAscii()
    animationRef.current = requestAnimationFrame(animate)
  }, [addParticles, convertToAscii])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    initMicAudio()

    const animTimeout = setTimeout(() => {
      if (canvasRef.current && asciiRef.current) {
        animate()
      }
    }, 100)

    return () => {
      window.removeEventListener("resize", resizeCanvas)
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      clearTimeout(animTimeout)
      disconnectMic()
      disconnectMp3()
      if (audioElementRef.current && audioElementRef.current.src && audioElementRef.current.src.startsWith("blob:")) {
        URL.revokeObjectURL(audioElementRef.current.src)
      }
      if (audioContextRef.current && audioContextRef.current.state !== "closed") {
        audioContextRef.current.close().catch((err) => console.warn("Error closing audio context", err))
      }
    }
  }, [initMicAudio, animate, disconnectMic, disconnectMp3])

  const getCurrentSourceValue = () => {
    if (currentAudioSourceType === "mic") {
      return "mic"
    } else if (currentPlayingSong) {
      return currentPlayingSong
    } else if (currentPlayingFile) {
      return currentPlayingFile.name
    }
    return "mic"
  }

  const getCurrentDisplayName = () => {
    if (currentAudioSourceType === "mic") {
      return "Microphone"
    } else if (currentPlayingSong) {
      return currentPlayingSong
    } else if (currentPlayingFile) {
      return currentPlayingFile.name
    }
    return "Microphone"
  }

  return (
    <div className="fixed inset-0 bg-black">
      <div className="fixed top-4 right-4 z-50 flex flex-col items-end space-y-2">
        <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
          <DialogTrigger asChild>
            <Button variant="outline" size="icon" className="bg-gray-800 hover:bg-gray-700 border-gray-600 text-white">
              <Settings className="h-5 w-5" />
              <span className="sr-only">Open Settings</span>
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-gray-900 text-white border-gray-700">
            <DialogHeader>
              <DialogTitle>Audio Settings</DialogTitle>
              <DialogDescription className="text-gray-400">
                Upload an MP3 file or select an audio source.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="mp3-upload" className="col-span-1 text-right">
                  Upload MP3
                </Label>
                <Input
                  id="mp3-upload"
                  type="file"
                  accept=".mp3"
                  onChange={handleFileChange}
                  className="col-span-3 bg-gray-800 border-gray-600 text-white file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:bg-gray-700 file:text-gray-300 hover:file:bg-gray-600"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="audio-source" className="col-span-1 text-right">
                  Source
                </Label>
                <Select
                  value={getCurrentSourceValue()}
                  onValueChange={handleAudioSourceSelect}
                >
                  <SelectTrigger className="col-span-3 bg-gray-800 border-gray-600 text-white">
                    <SelectValue placeholder="Select audio source" />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-800 text-white border-gray-700">
                    <SelectItem value="mic" className="hover:bg-gray-700 focus:bg-gray-700">
                      <div className="flex items-center">
                        <Mic className="mr-2 h-4 w-4" /> Microphone
                      </div>
                    </SelectItem>
                    {defaultSongs.map((song) => (
                      <SelectItem key={song.filename} value={song.displayName} className="hover:bg-gray-700 focus:bg-gray-700">
                        <div className="flex items-center">
                          <Music className="mr-2 h-4 w-4" /> {song.displayName}
                        </div>
                      </SelectItem>
                    ))}
                    {uploadedFiles.map((file) => (
                      <SelectItem key={file.name} value={file.name} className="hover:bg-gray-700 focus:bg-gray-700">
                        <div className="flex items-center">
                          <Music className="mr-2 h-4 w-4" /> {file.name}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <DialogFooter>
              <Button
                onClick={() => setIsSettingsOpen(false)}
                variant="outline"
                className="bg-gray-700 hover:bg-gray-600 border-gray-500 text-white"
              >
                Close
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {currentAudioSourceType === "mp3" && (currentPlayingFile || currentPlayingSong) && (
          <div className="bg-gray-800 p-3 rounded-lg shadow-lg w-72 border border-gray-700 text-white">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm font-medium truncate" title={getCurrentDisplayName()}>
                <Music className="inline mr-1 h-4 w-4" /> {getCurrentDisplayName()}
              </p>
              <Button variant="ghost" size="icon" onClick={togglePlayPause} className="hover:bg-gray-700 text-white">
                {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
              </Button>
            </div>
            <div className="flex items-center space-x-2 mb-1">
              <span className="text-xs w-10 text-gray-400">{formatTime(currentTime)}</span>
              <Input
                type="range"
                min="0"
                max={duration || 0}
                value={currentTime || 0}
                onChange={handleSeek}
                className="h-2 flex-grow appearance-none bg-gray-700 rounded-full cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-sky-400"
              />
              <span className="text-xs w-10 text-gray-400">{formatTime(duration)}</span>
            </div>
            <div className="flex items-center space-x-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  if (!audioElementRef.current) return
                  const newVolume = audioElementRef.current.volume > 0 ? 0 : 1
                  audioElementRef.current.volume = newVolume
                  setPlayerVolume(newVolume)
                }}
                className="hover:bg-gray-700 text-white"
              >
                {playerVolume > 0 ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
              </Button>
              <Input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={playerVolume}
                onChange={handleVolumeChange}
                className="h-2 flex-grow appearance-none bg-gray-700 rounded-full cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-sky-400"
              />
            </div>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} className="w-full h-full opacity-0" />
      <div
        ref={asciiRef}
        className="fixed inset-0 font-mono text-white whitespace-pre overflow-hidden pointer-events-none flex items-center justify-center"
        style={{ fontSize: "5px", lineHeight: "5px", letterSpacing: "-0.5px" }}
      />
    </div>
  )
}
