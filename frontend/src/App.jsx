import { useEffect } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { IconAlert, IconAssess, IconHistory, IconHome, IconInfo, IconPulse } from "./components/Icons.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Assess from "./pages/Assess.jsx";
import Result from "./pages/Result.jsx";
import ModelInfo from "./pages/ModelInfo.jsx";
import History from "./pages/History.jsx";

const NAV = [
  { to: "/", label: "Dashboard", icon: IconHome, end: true },
  { to: "/assess", label: "Assessment", icon: IconAssess },
  { to: "/model", label: "Model card", icon: IconInfo },
  { to: "/history", label: "History", icon: IconHistory }
];

function ResearchBanner() {
  return (
    <div className="research-banner" role="note">
      <IconAlert size={18} />
      <p>
        <strong>Research prototype.</strong> Not a medical device and not a substitute for
        clinical judgement. Never use these outputs to decide the mode of delivery.
      </p>
    </div>
  );
}

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [pathname]);
  return null;
}

function Footer() {
  return (
    <footer className="site-footer">
      <div>
        <p className="footer-title">OB-SIGHT · Delivery outcome research prototype</p>
        <p>
          Built on the public CTU-CHB Intrapartum Cardiotocography Database v1.0.0
          (PhysioNet). De-identified research data only. No external validation.
        </p>
        <p className="footer-meta">
          Version 1.1.0 · No data entered here is sent anywhere except the local
          decision-support API.
        </p>
      </div>
    </footer>
  );
}

export default function App() {
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <ScrollToTop />
      <header className="site-header">
        <div className="header-inner">
          <Link to="/" className="brand" aria-label="OB-SIGHT home">
            <span className="brand-mark">
              <IconPulse size={22} />
            </span>
            <span className="brand-text">
              <span className="brand-name">OB-SIGHT</span>
              <span className="brand-sub">Delivery outcome CDSS · research</span>
            </span>
          </Link>
          <nav className="top-nav" aria-label="Primary">
            {NAV.map(({ to, label, end }) => (
              <NavLink key={to} to={to} end={end} className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
                {label}
              </NavLink>
            ))}
          </nav>
          <span className="version-pill" title="Prototype version">
            v1.1
          </span>
        </div>
      </header>
      <ResearchBanner />
      <main id="main" className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/assess" element={<Assess />} />
          <Route path="/result" element={<Result />} />
          <Route path="/model" element={<ModelInfo />} />
          <Route path="/history" element={<History />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <Footer />
      <nav className="bottom-nav" aria-label="Primary mobile">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} className={({ isActive }) => (isActive ? "bottom-link active" : "bottom-link")}>
            <Icon size={21} />
            <span>{label === "Model card" ? "Model" : label}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  );
}

function NotFound() {
  return (
    <section className="page narrow">
      <div className="empty-state card">
        <h1>Page not found</h1>
        <p>The page you requested does not exist in this prototype.</p>
        <Link className="btn btn-primary" to="/">
          Back to dashboard
        </Link>
      </div>
    </section>
  );
}
