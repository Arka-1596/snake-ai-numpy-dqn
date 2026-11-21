
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Snake Studies — The Art of Learning",
  description:
    "An interactive exploration of Snake AI, neural networks, and reinforcement learning built from scratch with NumPy.",
  keywords: [
    "Snake AI",
    "Deep Q-Learning",
    "Reinforcement Learning",
    "NumPy",
    "Neural Network",
    "Interactive Art",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
