import type { Metadata } from 'next'
import Link from 'next/link'
import './globals.css'

export const metadata: Metadata = {
  title: 'Music Visualizer',
  description: 'Visualize music with GLSL',
  generator: 'Your Mom',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>
        
            <div className="flex gap-4 bg-white text-black shadow-lg fixed top-0 left-0 right-0 z-50">
              <Link 
                href="/" 
                className="hover:text-blue-300 transition-colors duration-200 font-medium"
              >
                Home
              </Link>
              <Link 
                href="/ball" 
                className="hover:text-blue-300 transition-colors duration-200 font-medium"
              >
                Ball
              </Link>
              <Link 
                href="/fractal" 
                className="hover:text-blue-300 transition-colors duration-200 font-medium"
              >
                Fractal
              </Link>
              <Link 
                href="/inertia" 
                className="hover:text-blue-300 transition-colors duration-200 font-medium"
              >
                Inertia
              </Link>
              <Link 
                href="/geometric" 
                className="hover:text-blue-300 transition-colors duration-200 font-medium"
              >
                Geometric
              </Link>
              <Link 
                href="/crystal" 
                className="hover:text-blue-300 transition-colors duration-200 font-medium"
              >
                Crystal
              </Link>
              <Link 
                href="/kaleidoscope" 
                className="hover:text-blue-300 transition-colors duration-200 font-medium"
              >
                Kaleidoscope
              </Link>
              <Link 
                href="/snow" 
                className="hover:text-blue-300 transition-colors duration-200 font-medium"
              >
                Snow
              </Link>
              <Link 
                href="/starwars" 
                className="hover:text-blue-300 transition-colors duration-200 font-medium"
              >
                Star Wars
              </Link>
              <Link 
                href="/tunnel" 
                className="hover:text-blue-300 transition-colors duration-200 font-medium"
              >
                Tunnel
              </Link>
            </div>
        {children}
      </body>
    </html>
  )
}
