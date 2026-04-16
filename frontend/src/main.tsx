import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App"; // SteelVisionApp 대신 원본 App.tsx 사용
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
