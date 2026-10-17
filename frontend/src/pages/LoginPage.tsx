import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import {
  Mail,
  Lock,
  Eye,
  EyeOff,
  Layers,
  ArrowRight,
  ShieldCheck,
  AlertCircle,
  Cpu,
} from 'lucide-react';

export const LoginPage: React.FC = () => {
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [email, setEmail] = useState('priya.m@medai.org');
  const [password, setPassword] = useState('medai2026');
  const [rememberMe, setRememberMe] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // If already logged in, redirect to dashboard or prior page
  React.useEffect(() => {
    if (isAuthenticated) {
      const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/dashboard';
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, navigate, location]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);

    if (!email || !password) {
      setErrorMessage('Please provide both institutional email and security credentials.');
      return;
    }

    setIsLoading(true);
    try {
      await login({ email, password, rememberMe });
      const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/dashboard';
      navigate(from, { replace: true });
    } catch (err: unknown) {
      if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Authentication failed. Please verify credentials.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickFill = (demoEmail: string) => {
    setEmail(demoEmail);
    setPassword('medai2026');
    setErrorMessage(null);
  };

  return (
    <div className="min-h-screen w-full flex bg-[#F8FAFC] text-slate-800 antialiased overflow-x-hidden">
      {/* ======================================================== */}
      {/* LEFT PANEL: Brand & Research Identity (~45% Desktop)     */}
      {/* ======================================================== */}
      <div className="hidden lg:flex lg:w-[45%] xl:w-[42%] relative bg-[#091322] flex-col justify-between p-8 xl:p-12 overflow-hidden border-r border-slate-800/80 shrink-0">
        {/* Subtle grid pattern and soft background ambient glow */}
        <div className="absolute inset-0 bg-[radial-gradient(#1E3A8A_1px,transparent_1px)] [background-size:24px_24px] opacity-20 pointer-events-none" />
        <div className="absolute top-1/4 -left-12 w-80 h-80 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-1/4 right-0 w-80 h-80 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

        {/* Top: MEDAI / RESEARCH LAB Branding */}
        <div className="relative z-10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-cyan-500 p-[1.5px] shadow-sm">
              <div className="w-full h-full bg-[#091322] rounded-[10px] flex items-center justify-center">
                <Layers className="w-5 h-5 text-cyan-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-extrabold tracking-wider text-white">
                  MED<span className="text-cyan-400">AI</span>
                </span>
                <span className="text-[10px] uppercase font-bold tracking-widest px-2 py-0.5 rounded bg-blue-500/15 text-blue-300 border border-blue-500/25">
                  Research Lab
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-medium tracking-wide">
                Medical AI Research Platform
              </p>
            </div>
          </div>
        </div>

        {/* Center: Research Narrative & Highlights */}
        <div className="relative z-10 my-auto py-6">
          {/* Research Badge */}
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-950/70 border border-blue-800/60 text-cyan-300 text-[11px] font-semibold tracking-wide mb-5">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            <span>TRANSFER LEARNING &bull; MEDICAL IMAGING</span>
          </div>

          {/* Headlines */}
          <h1 className="text-2xl xl:text-3xl 2xl:text-4xl font-extrabold text-white tracking-tight leading-[1.2] mb-3">
            Intelligent Medical Imaging.<br />
            <span className="bg-gradient-to-r from-blue-400 via-cyan-300 to-indigo-300 bg-clip-text text-transparent">
              Powered by Deep Learning.
            </span>
          </h1>

          <p className="text-xs xl:text-sm text-slate-300 leading-relaxed max-w-md mb-6">
            Standardized comparative evaluation of deep transfer learning architectures for pediatric chest radiograph screening on PneumoniaMNIST.
          </p>

          {/* Three Compact Research Highlights (01, 02, 03) */}
          <div className="space-y-2.5 mb-6 max-w-md">
            <div className="flex items-start gap-3 p-2.5 rounded-xl bg-slate-900/60 border border-slate-800/80">
              <span className="text-[10px] font-mono font-bold text-cyan-400 px-1.5 py-0.5 rounded bg-cyan-950/70 border border-cyan-800/40 shrink-0">
                01
              </span>
              <div>
                <h2 className="text-xs font-bold text-white">Transfer Learning</h2>
                <p className="text-[11px] text-slate-400 leading-normal">
                  Pre-trained ImageNet representations fine-tuned on standardized chest radiographs.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-2.5 rounded-xl bg-slate-900/60 border border-slate-800/80">
              <span className="text-[10px] font-mono font-bold text-blue-400 px-1.5 py-0.5 rounded bg-blue-950/70 border border-blue-800/40 shrink-0">
                02
              </span>
              <div>
                <h2 className="text-xs font-bold text-white">Medical Image Classification</h2>
                <p className="text-[11px] text-slate-400 leading-normal">
                  Objective binary classification between Normal and Pneumonia pediatric cohorts.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-2.5 rounded-xl bg-slate-900/60 border border-slate-800/80">
              <span className="text-[10px] font-mono font-bold text-indigo-400 px-1.5 py-0.5 rounded bg-indigo-950/70 border border-indigo-800/40 shrink-0">
                03
              </span>
              <div>
                <h2 className="text-xs font-bold text-white">Model Explainability</h2>
                <p className="text-[11px] text-slate-400 leading-normal">
                  Grad-CAM gradient visual attention maps highlighting decisive radiographic regions.
                </p>
              </div>
            </div>
          </div>

          {/* Small Model Comparison Section */}
          <div className="grid grid-cols-2 gap-3 max-w-md">
            <div className="p-3 rounded-xl bg-slate-900/80 border border-blue-900/40">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-bold text-white flex items-center gap-1.5">
                  <Cpu className="w-3 h-3 text-blue-400" />
                  RESNET50
                </span>
                <span className="text-[10px] font-mono text-blue-300 font-semibold">Residual</span>
              </div>
              <p className="text-[10px] text-slate-400">Deep residual bottlenecks</p>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-cyan-900/40">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-bold text-white flex items-center gap-1.5">
                  <Cpu className="w-3 h-3 text-cyan-400" />
                  EFFICIENTNET-B0
                </span>
                <span className="text-[10px] font-mono text-cyan-300 font-semibold">MBConv</span>
              </div>
              <p className="text-[10px] text-slate-400">Compound scaled MBConv</p>
            </div>
          </div>
        </div>

        {/* Bottom Status Area */}
        <div className="relative z-10 pt-5 border-t border-slate-800/80">
          <div className="flex items-center justify-between text-[11px]">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="font-semibold text-slate-300">Research Environment</span>
            </div>
            <span className="text-slate-400 font-medium">Status: Ready for analysis</span>
          </div>
          <div className="flex items-center gap-3 mt-2 text-[10px] text-slate-500 font-mono">
            <span>Dataset: <span className="text-slate-300">PneumoniaMNIST</span></span>
            <span>&bull;</span>
            <span>Framework: <span className="text-slate-300">PyTorch</span></span>
            <span>&bull;</span>
            <span>Device: <span className="text-slate-300">CPU</span></span>
          </div>
        </div>
      </div>

      {/* ======================================================== */}
      {/* RIGHT PANEL: Clean Authentication Card (~55% Desktop)    */}
      {/* ======================================================== */}
      <div className="w-full lg:w-[55%] xl:w-[58%] min-h-screen flex flex-col justify-center items-center px-4 sm:px-8 md:px-12 py-8 sm:py-12 relative">
        <div className="w-full max-w-[420px]">
          {/* Mobile brand header (shown only on mobile/tablet when left panel is hidden) */}
          <div className="lg:hidden mb-6 sm:mb-8">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-xl bg-slate-900 flex items-center justify-center text-cyan-400 shadow-sm shrink-0">
                <Layers className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-base font-extrabold text-slate-900 tracking-wider">
                    MED<span className="text-blue-600">AI</span>
                  </span>
                  <span className="text-[10px] uppercase font-bold tracking-widest px-1.5 py-0.5 rounded bg-blue-100 text-blue-700">
                    Research Lab
                  </span>
                </div>
                <p className="text-[11px] text-slate-500">Medical AI Research Platform</p>
              </div>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Transfer Learning &bull; PneumoniaMNIST &bull; ResNet50 vs EfficientNetB0
            </p>
          </div>

          {/* Login Card */}
          <div className="bg-white rounded-2xl p-6 sm:p-8 shadow-sm border border-slate-200/90 relative">
            <div className="mb-6">
              <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">
                Welcome back
              </h2>
              <p className="text-xs sm:text-sm text-slate-500 mt-1.5 leading-relaxed">
                Sign in to continue to the Medical AI Research Platform.
              </p>
            </div>

            {/* Error Message if any */}
            {errorMessage && (
              <div className="mb-5 p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-start gap-2.5 animate-in fade-in duration-200">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                <div className="flex-1 font-medium leading-relaxed">{errorMessage}</div>
              </div>
            )}

            {/* Sign In Form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Email field */}
              <div>
                <label
                  htmlFor="email-input"
                  className="block text-xs font-semibold text-slate-700 mb-1.5"
                >
                  Email address
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                    <Mail className="w-4 h-4" />
                  </div>
                  <input
                    id="email-input"
                    type="email"
                    required
                    autoComplete="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="researcher@medai.org"
                    className="block w-full pl-10 pr-3.5 py-2.5 sm:py-3 bg-slate-50/70 border border-slate-200 rounded-xl text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600/25 focus:border-blue-600 focus:bg-white transition-all min-h-[44px]"
                  />
                </div>
              </div>

              {/* Password field */}
              <div>
                <label
                  htmlFor="password-input"
                  className="block text-xs font-semibold text-slate-700 mb-1.5"
                >
                  Password
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                    <Lock className="w-4 h-4" />
                  </div>
                  <input
                    id="password-input"
                    type={showPassword ? 'text' : 'password'}
                    required
                    autoComplete="current-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="block w-full pl-10 pr-11 py-2.5 sm:py-3 bg-slate-50/70 border border-slate-200 rounded-xl text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600/25 focus:border-blue-600 focus:bg-white transition-all font-mono min-h-[44px]"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-slate-400 hover:text-slate-600 focus:outline-none min-w-[44px] justify-end"
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {/* Remember Me and Forgot Password */}
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs pt-1">
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    className="w-4 h-4 rounded text-blue-600 border-slate-300 focus:ring-blue-500 focus:ring-offset-0 transition-colors"
                  />
                  <span className="text-slate-600 font-medium">Remember me</span>
                </label>

                <button
                  type="button"
                  onClick={() => alert('For password resets on the research platform, contact the local lab administrator.')}
                  className="text-blue-600 hover:text-blue-700 font-medium hover:underline transition-colors"
                >
                  Forgot password?
                </button>
              </div>

              {/* Sign In Button */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full min-h-[48px] flex items-center justify-center gap-2 py-3 px-4 rounded-xl text-sm font-semibold text-white bg-slate-900 hover:bg-blue-700 active:bg-blue-800 disabled:opacity-60 disabled:cursor-not-allowed shadow-sm hover:shadow-md transition-all duration-200 mt-2"
              >
                {isLoading ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Signing in...</span>
                  </>
                ) : (
                  <>
                    <span>Sign in</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </form>

            {/* Quick Research Access (Preserved functionality with subtle, professional design) */}
            <div className="mt-6 pt-5 border-t border-slate-100">
              <div className="flex items-center justify-between text-xs text-slate-500 mb-2.5">
                <span className="font-semibold text-slate-700">Quick research access</span>
                <span className="text-[11px] text-slate-400">Autofill credentials</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => handleQuickFill('priya.m@medai.org')}
                  className="px-3 py-2 text-xs font-medium text-slate-700 bg-slate-50 hover:bg-slate-100 hover:text-slate-900 border border-slate-200/80 rounded-lg transition-colors flex items-center justify-between min-h-[40px]"
                  title="Autofill Lead Researcher"
                >
                  <span>Priya M (Researcher)</span>
                  <span className="text-[10px] text-slate-400 font-mono">Fill &rarr;</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleQuickFill('admin@medai.org')}
                  className="px-3 py-2 text-xs font-medium text-slate-700 bg-slate-50 hover:bg-slate-100 hover:text-slate-900 border border-slate-200/80 rounded-lg transition-colors flex items-center justify-between min-h-[40px]"
                  title="Autofill Administrator"
                >
                  <span>Administrator</span>
                  <span className="text-[10px] text-slate-400 font-mono">Fill &rarr;</span>
                </button>
              </div>
            </div>

            {/* Subtle Research Disclaimer */}
            <div className="mt-6 pt-4 border-t border-slate-100 text-center">
              <p className="text-[11px] text-slate-400 leading-normal flex items-center justify-center gap-1.5 font-medium">
                <ShieldCheck className="w-3.5 h-3.5 text-blue-500 shrink-0" />
                <span>Educational Research Platform — Not for Clinical Use</span>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
