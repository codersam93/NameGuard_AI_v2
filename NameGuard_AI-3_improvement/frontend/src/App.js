import { BrowserRouter, Routes, Route } from "react-router-dom";
import "@/App.css";
import NameEvaluatorPage from "@/pages/NameEvaluator";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<NameEvaluatorPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
