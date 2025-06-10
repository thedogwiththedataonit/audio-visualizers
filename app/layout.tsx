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
        <header className="bg-gray-900 text-white shadow-lg absolute top-0 left-0 right-0 z-50">
          <nav className="container mx-auto px-4 py-4">
            <div className="flex gap-4">
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
            </div>
          </nav>
        </header>
        {children}
      </body>
    </html>
  )
}
