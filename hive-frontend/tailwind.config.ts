import type { Config } from "tailwindcss";
export default { content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"], theme: { extend: { colors: { void: "#070e18", panel: "#0c1926", edge: "#23374c", mars: "#f18754" } } }, plugins: [] } satisfies Config;
