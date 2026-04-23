// Default React namespace import required for JSX runtime (<StrictMode>, etc.).
import React from "react";
// Client-side React 18 root API for mounting the SPA into index.html’s #root div.
import ReactDOM from "react-dom/client";
// Root application component containing layout, upload, chat, and sources UI.
import App from "./App.jsx";
// Global stylesheet: fonts, CSS variables, base element resets.
import "./index.css";

// Create a concurrent React root attached to the DOM element with id "root".
ReactDOM.createRoot(document.getElementById("root")).render(
  // StrictMode enables extra development checks (warnings, double-invoke effects).
  <React.StrictMode>
    {/* App is the top-level routed-less SPA shell for this project. */}
    <App />
  </React.StrictMode>,
);
