"use client"

import { useState } from "react"

interface ColorTheme {
  name: string
  baseHue: number
  saturation: number
}

interface ColorThemeSelectorProps {
  onSelectTheme: (theme: ColorTheme) => void
}

const themes: ColorTheme[] = [
  { name: "Default", baseHue: 0.08, saturation: 0.8 },
  { name: "Cool Blue", baseHue: 0.6, saturation: 0.9 },
  { name: "Neon Green", baseHue: 0.3, saturation: 1.0 },
  { name: "Sunset", baseHue: 0.05, saturation: 0.85 },
  { name: "Purple Haze", baseHue: 0.75, saturation: 0.7 },
  // New color themes
  { name: "Cyber Pink", baseHue: 0.9, saturation: 0.95 },
  { name: "Electric Teal", baseHue: 0.45, saturation: 0.9 },
  { name: "Golden Hour", baseHue: 0.12, saturation: 0.85 },
  { name: "Deep Ocean", baseHue: 0.55, saturation: 0.8 },
  { name: "Midnight", baseHue: 0.65, saturation: 0.7 },
  { name: "Lava Flow", baseHue: 0.02, saturation: 0.95 },
  { name: "Forest", baseHue: 0.35, saturation: 0.75 },
]

export default function ColorThemeSelector({ onSelectTheme }: ColorThemeSelectorProps) {
  const [selectedTheme, setSelectedTheme] = useState(themes[0])

  const handleSelectTheme = (theme: ColorTheme) => {
    setSelectedTheme(theme)
    onSelectTheme(theme)
  }

  return (
    <div className="w-full">
      {/* Current theme display */}
      <div className="mb-4 p-3 bg-black/40 rounded-lg">
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-full"
            style={{
              backgroundColor: `hsl(${selectedTheme.baseHue * 360}deg ${selectedTheme.saturation * 100}% 50%)`,
            }}
          />
          <div>
            <div className="font-medium">{selectedTheme.name}</div>
            <div className="text-xs text-gray-300">Currently selected</div>
          </div>
        </div>
      </div>

      {/* Theme grid */}
      <div className="grid grid-cols-2 gap-3">
        {themes.map((theme) => (
          <button
            key={theme.name}
            onClick={() => handleSelectTheme(theme)}
            className={`flex flex-col items-center p-3 rounded-lg transition-all ${
              selectedTheme.name === theme.name ? "bg-white/20 ring-1 ring-white/40" : "bg-black/40 hover:bg-white/10"
            }`}
          >
            <div
              className="w-10 h-10 rounded-full mb-2"
              style={{
                backgroundColor: `hsl(${theme.baseHue * 360}deg ${theme.saturation * 100}% 50%)`,
              }}
            />
            <span className="text-sm text-center">{theme.name}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
