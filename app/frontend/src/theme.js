import { useEffect, useState } from "react";

// Chart colors keyed by strategy + a shared series palette. Mid-tones chosen to
// read on both light and dark backgrounds.
export const STRAT_COLORS = {
  join_touch: "#2d77d4",
  fixed_spread_1t: "#bf8521",
  avellaneda_stoikov: "#d94759",
  bayesian_queue: "#0e9f70",
  rl_dqn: "#6f52c9",
};
export const SERIES = ["#2d77d4", "#0e9f70", "#bf8521", "#6f52c9", "#d94759", "#1fa8a8"];

export function niceName(s) {
  return s.replace(/_/g, " ").replace(/\b1t\b/, "(1 tick)");
}

export function useTheme() {
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("qamm-theme") || "system"; } catch { return "system"; }
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try { localStorage.setItem("qamm-theme", theme); } catch {}
  }, [theme]);
  const cycle = () => setTheme((t) => (t === "system" ? "light" : t === "light" ? "dark" : "system"));
  return [theme, cycle];
}

// read current token values so Recharts axes/grid follow the theme
export function useTokens(dep) {
  const [tok, setTok] = useState({ muted: "#5a6a86", line: "#dde4f0", ink: "#131b2d" });
  useEffect(() => {
    const cs = getComputedStyle(document.documentElement);
    const get = (n, f) => (cs.getPropertyValue(n).trim() || f);
    setTok({ muted: get("--muted", "#5a6a86"), line: get("--line", "#dde4f0"), ink: get("--ink", "#131b2d") });
  }, [dep]);
  return tok;
}
