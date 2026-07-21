import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles/globals.css";

// Dev-only preview harness for the StyledTextEditor — accessible with
// ?editor-test in the URL. Lets us visually verify centering and
// selection alignment without booting the Tauri config pipeline.
const isEditorTest =
  typeof window !== "undefined" &&
  window.location.search.includes("editor-test");

const render = (el: React.ReactElement) =>
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>{el}</React.StrictMode>,
  );

if (isEditorTest) {
  import("./components/grid/StyledTextEditorTest").then(({ EditorTest }) => {
    render(<EditorTest />);
  });
} else {
  render(<App />);
}
