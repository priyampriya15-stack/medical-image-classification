import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Database,
  Cpu,
  GitCompare,
  ScanLine,
  Flame,
  LineChart,
  Info,
  ChevronLeft,
  ChevronRight,
  ShieldAlert,
  X,
} from 'lucide-react';

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

interface NavItem {
  name: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
  aliases?: string[];
}

const NAV_ITEMS: NavItem[] = [
  { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
  { name: 'Dataset Explorer', path: '/dataset', icon: Database, badge: '5.8k' },
  { name: 'Model Training', path: '/training', icon: Cpu },
  { name: 'Model Comparison', path: '/model-comparison', icon: GitCompare, aliases: ['/comparison'] },
  { name: 'Image Classifier', path: '/classifier', icon: ScanLine },
  { name: 'Grad-CAM', path: '/gradcam', icon: Flame },
  { name: 'Training Analytics', path: '/training-analytics', icon: LineChart, aliases: ['/analytics'] },
  { name: 'Project Information', path: '/project-info', icon: Info },
];

export const Sidebar: React.FC<SidebarProps> = ({
  collapsed,
  onToggleCollapse,
  mobileOpen,
  onCloseMobile,
}) => {
  const location = useLocation();
  return (
    <>
      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/60 backdrop-blur-xs lg:hidden transition-opacity"
          onClick={onCloseMobile}
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 flex flex-col bg-[#0B1528] text-slate-200 border-r border-slate-800 transition-all duration-300 ease-in-out
          ${collapsed ? 'w-20' : 'w-64 max-w-[85vw]'}
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Brand Header */}
        {/* Brand Header */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-slate-800/80 bg-[#0B1528]">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-500 to-cyan-400 p-[1.5px] shrink-0 shadow-glow-blue flex items-center justify-center">
              <div className="w-full h-full bg-[#0B1528] rounded-[10px] flex items-center justify-center">
                {/* Glowing Brain / Neural Icon matching screenshot */}
                <svg className="w-5 h-5 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2a4 4 0 0 0-4 4v1a4 4 0 0 0-4 4 4 4 0 0 0 2 3.5A4 4 0 0 0 6 18a4 4 0 0 0 6 4 4 4 0 0 0 6-4 4 4 0 0 0 0-3.5 4 4 0 0 0 2-3.5 4 4 0 0 0-4-4V6a4 4 0 0 0-4-4z" />
                  <path d="M12 2v20" />
                  <path d="M8 8a3 3 0 0 0 3-3" />
                  <path d="M16 8a3 3 0 0 1-3-3" />
                  <path d="M6 13a3 3 0 0 1 3-3" />
                  <path d="M18 13a3 3 0 0 0-3-3" />
                  <path d="M8 18a3 3 0 0 0 3-3" />
                  <path d="M16 18a3 3 0 0 1-3-3" />
                </svg>
              </div>
            </div>

            {!collapsed && (
              <div className="flex flex-col truncate">
                <span className="font-extrabold text-base tracking-wide text-white">
                  MedAI
                </span>
                <span className="text-[11px] text-slate-400 font-medium truncate">
                  Research Platform
                </span>
              </div>
            )}
          </div>

          {/* Close button on mobile */}
          <button
            onClick={onCloseMobile}
            className="min-w-[44px] min-h-[44px] p-2.5 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 lg:hidden flex items-center justify-center"
            title="Close navigation"
            aria-label="Close navigation"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation List */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1.5">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isMatch = location.pathname === item.path || (item.aliases && item.aliases.includes(location.pathname));
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={onCloseMobile}
                title={collapsed ? item.name : undefined}
                className={({ isActive }) => {
                  const active = isActive || Boolean(isMatch);
                  return `group relative flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                    active
                      ? 'bg-[#2563EB] text-white shadow-md font-semibold'
                      : 'text-slate-300 hover:text-white hover:bg-slate-800/70'
                  } ${collapsed ? 'justify-center' : ''}`;
                }}
              >
                {({ isActive }) => {
                  const active = isActive || Boolean(isMatch);
                  return (
                    <>
                      <Icon
                        className={`w-4 h-4 shrink-0 transition-transform group-hover:scale-105 ${
                          active ? 'text-white' : 'text-slate-400 group-hover:text-slate-200'
                        }`}
                      />

                      {!collapsed && (
                        <div className="flex-1 flex items-center justify-between truncate">
                          <span className="truncate">{item.name}</span>
                          {item.badge && (
                            <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-md ${active ? 'bg-blue-700/80 text-blue-100' : 'bg-slate-800 text-slate-400'}`}>
                              {item.badge}
                            </span>
                          )}
                        </div>
                      )}
                    </>
                  );
                }}
              </NavLink>
            );
          })}
        </nav>

        {/* Bottom Disclaimer / Footer info matching screenshot */}
        <div className="p-4 border-t border-slate-800/80 bg-[#08101E]">
          {!collapsed ? (
            <div className="space-y-2">
              {/* Lungs outline illustration */}
              <div className="w-10 h-10 rounded-xl bg-blue-950/60 border border-blue-800/40 flex items-center justify-center text-cyan-400 mb-2">
                <svg className="w-6 h-6 text-cyan-400 drop-shadow-[0_0_8px_rgba(34,211,238,0.5)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 4v8" />
                  <path d="M12 9c-2-2-4.5-3-7-3A4 4 0 0 0 1 10c0 5 4 10 9 10h2c5 0 9-5 9-10a4 4 0 0 0-4-4c-2.5 0-5 1-7 3" />
                  <circle cx="6" cy="14" r="1" fill="currentColor" />
                  <circle cx="18" cy="14" r="1" fill="currentColor" />
                </svg>
              </div>

              <div>
                <h4 className="text-xs font-bold text-white tracking-tight">
                  AI for Better Healthcare
                </h4>
                <p className="text-[11px] text-slate-400 leading-relaxed mt-0.5">
                  Leveraging deep learning for medical image classification and analysis.
                </p>
              </div>

              <div className="pt-2 border-t border-slate-800/60 text-[10px] text-slate-500 font-medium">
                Educational Project &bull; Not for clinical use.
              </div>
            </div>
          ) : (
            <div className="flex justify-center" title="AI for Better Healthcare - Educational Project">
              <ShieldAlert className="w-4 h-4 text-cyan-400" />
            </div>
          )}

          {/* Desktop Collapse Toggle */}
          <div className="hidden lg:flex items-center justify-end mt-2 pt-1 border-t border-slate-800/40">
            <button
              onClick={onToggleCollapse}
              className="w-full py-1.5 px-2 flex items-center justify-center gap-2 rounded-lg text-xs text-slate-400 hover:text-white hover:bg-slate-800/80 transition-colors"
              title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              {collapsed ? (
                <ChevronRight className="w-4 h-4" />
              ) : (
                <>
                  <ChevronLeft className="w-4 h-4" />
                  <span className="text-[11px]">Collapse sidebar</span>
                </>
              )}
            </button>
          </div>
        </div>
      </aside>
    </>
  );
};
