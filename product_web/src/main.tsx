import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import Compare from "./Compare";
import "./style.css";
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {new URLSearchParams(location.search).has("comparison") || new URLSearchParams(location.search).get("mode") === "compare" ? <Compare /> : <App />}
  </StrictMode>,
);
