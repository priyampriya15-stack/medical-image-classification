import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '../../auth/AuthContext';
import { useNavigate } from 'react-router-dom';
import {
  Menu,
  Sun,
  Moon,
  LogOut,
  User as UserIcon,
  Settings,
  ChevronDown,
  ShieldCheck,
  Image as ImageIcon,
} from 'lucide-react';
import { medicalApi } from '../../services/api';
import type { BackendHealth } from '../../types';

interface HeaderProps {
  onToggleMobileMenu: () => void;
  collapsed: boolean;
}

export const Header: React.FC<HeaderProps> = ({ onToggleMobileMenu, collapsed }) => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [backendHealth, setBackendHealth] = useState<BackendHealth>({
    isConnected: false,
    latencyMs: 0,
    serverUrl: '',
  });

  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Poll/check backend health on mount
    medicalApi.checkHealth().then((health) => {
      setBackendHealth(health);
    });

    const interval = setInterval(() => {
      medicalApi.checkHealth().then(setBackendHealth);
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  // Close dropdown on click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = async () => {
    setDropdownOpen(false);
    await logout();
    navigate('/login');
  };

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
    document.documentElement.classList.toggle('dark');
  };

  return (
    <header className={`sticky top-0 z-30 h-16 bg-white/95 backdrop-blur-md border-b border-slate-200/80 transition-all duration-300 ${collapsed ? 'lg:pl-20' : 'lg:pl-64'}`}>
      <div className="h-full px-3 sm:px-6 flex items-center justify-between gap-2">
        {/* Left: Mobile hamburger & Titles */}
        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
          <button
            onClick={onToggleMobileMenu}
            className="min-w-[44px] min-h-[44px] p-2 rounded-xl text-slate-600 hover:text-slate-900 hover:bg-slate-100 lg:hidden transition-colors flex items-center justify-center shrink-0"
            aria-label="Open navigation menu"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div className="flex items-center gap-3 min-w-0">
            {/* Image icon in rounded square matching screenshot */}
            <div className="hidden sm:flex w-10 h-10 rounded-xl bg-blue-50 border border-blue-200/80 items-center justify-center text-blue-600 shrink-0 shadow-2xs">
              <ImageIcon className="w-5 h-5 text-blue-600" />
            </div>

            <div className="min-w-0">
              <h1 className="font-bold text-slate-900 tracking-tight leading-tight truncate text-base sm:text-lg">
                Medical Image Classification
              </h1>
              <p className="text-xs text-slate-500 hidden sm:block truncate font-normal">
                ResNet50 vs EfficientNetB0 &nbsp;|&nbsp; Transfer Learning Research Platform
              </p>
            </div>
          </div>
        </div>

        {/* Right side tools */}
        <div className="flex items-center gap-2 sm:gap-3 shrink-0">
          {/* Backend Status indicator */}
          <div
            className={`flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-1 rounded-full text-xs font-semibold border transition-colors shrink-0 shadow-2xs ${
              backendHealth.isConnected
                ? 'bg-emerald-50 text-emerald-700 border-emerald-300'
                : 'bg-rose-50 text-rose-700 border-rose-300'
            }`}
            title={`Server: ${backendHealth.serverUrl}`}
          >
            <span className="relative flex h-2 w-2 shrink-0">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${backendHealth.isConnected ? 'bg-emerald-400 opacity-75' : 'bg-rose-400 opacity-75'}`} />
              <span className={`relative inline-flex rounded-full h-2 w-2 ${backendHealth.isConnected ? 'bg-emerald-500' : 'bg-rose-500'}`} />
            </span>
            <span className="hidden sm:inline">
              {backendHealth.isConnected ? 'Backend Connected' : 'Backend Disconnected'}
            </span>
            <span className="sm:hidden text-[10px] font-semibold uppercase">
              {backendHealth.isConnected ? 'Connected' : 'Offline'}
            </span>
          </div>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="w-9 h-9 p-2 rounded-xl text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition-colors flex items-center justify-center shrink-0 border border-slate-200/70"
            title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            aria-label="Toggle theme"
          >
            {isDarkMode ? <Sun className="w-4 h-4 text-amber-500" /> : <Moon className="w-4 h-4 text-slate-600" />}
          </button>

          {/* User profile dropdown - PM (Priya M) */}
          <div className="relative shrink-0" ref={dropdownRef}>
            <button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center gap-2 p-1 sm:px-2 sm:py-1 rounded-xl hover:bg-slate-100 transition-colors min-h-[40px]"
              aria-expanded={dropdownOpen}
              aria-label="User profile menu"
            >
              {/* Initials circle PM avatar */}
              <div className="w-8 h-8 rounded-full bg-[#0B1528] text-white flex items-center justify-center font-bold text-xs shadow-xs shrink-0 ring-2 ring-blue-500/20">
                PM
              </div>

              <span className="text-xs font-semibold text-slate-900 hidden sm:inline">
                {user?.name || 'Priya M'}
              </span>

              <ChevronDown className="w-3.5 h-3.5 text-slate-400 hidden sm:block" />
            </button>

            {/* Dropdown Menu */}
            {dropdownOpen && (
              <div className="absolute right-0 mt-2 w-56 max-w-[calc(100vw-1.5rem)] bg-white rounded-2xl shadow-card-hover border border-slate-200/90 py-1.5 z-50 animate-in fade-in slide-in-from-top-2 duration-150">
                <div className="px-4 py-3 border-b border-slate-100">
                  <p className="text-xs font-bold text-slate-900 truncate">
                    {user?.name || 'Dr. Elena Vance'}
                  </p>
                  <p className="text-[11px] text-slate-500 truncate">
                    {user?.email || 'researcher@medai.org'}
                  </p>
                  <div className="mt-1.5 flex items-center gap-1 text-[10px] text-blue-700 bg-blue-50 px-2 py-0.5 rounded-md font-medium border border-blue-100">
                    <ShieldCheck className="w-3 h-3 text-blue-600" />
                    <span>{user?.institution || 'Biomedical AI Lab'}</span>
                  </div>
                </div>

                <div className="py-1">
                  <button
                    onClick={() => {
                      setDropdownOpen(false);
                      navigate('/project-info');
                    }}
                    className="w-full flex items-center gap-2.5 px-4 py-2 text-xs text-slate-700 hover:bg-slate-50 hover:text-blue-600 transition-colors"
                  >
                    <UserIcon className="w-4 h-4 text-slate-400" />
                    <span>Researcher Profile</span>
                  </button>

                  <button
                    onClick={() => {
                      setDropdownOpen(false);
                      navigate('/project-info');
                    }}
                    className="w-full flex items-center gap-2.5 px-4 py-2 text-xs text-slate-700 hover:bg-slate-50 hover:text-blue-600 transition-colors"
                  >
                    <Settings className="w-4 h-4 text-slate-400" />
                    <span>Platform Settings</span>
                  </button>
                </div>

                <div className="pt-1 border-t border-slate-100">
                  <button
                    onClick={handleLogout}
                    className="w-full flex items-center gap-2.5 px-4 py-2 text-xs text-rose-600 hover:bg-rose-50 transition-colors font-medium"
                  >
                    <LogOut className="w-4 h-4 text-rose-500" />
                    <span>Sign Out</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
