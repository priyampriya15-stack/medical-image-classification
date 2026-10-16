import React, { useState } from 'react';
import { Outlet, useLocation, Link } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { ChevronRight, Home, ShieldAlert } from 'lucide-react';

export const AppLayout: React.FC = () => {
  const [collapsed, setCollapsed] = useState<boolean>(false);
  const [mobileOpen, setMobileOpen] = useState<boolean>(false);
  const location = useLocation();

  // Route to readable title mapping for breadcrumbs
  const pathnames = location.pathname.split('/').filter((x) => x);
  const getBreadcrumbTitle = (path: string) => {
    switch (path) {
      case 'dashboard':
        return 'Dashboard';
      case 'dataset':
        return 'Dataset Explorer';
      case 'training':
        return 'Model Training';
      case 'comparison':
        return 'Model Comparison';
      case 'classifier':
        return 'Image Classifier';
      case 'gradcam':
        return 'Grad-CAM Explainability';
      case 'analytics':
        return 'Training Analytics';
      case 'project-info':
        return 'Project Information';
      default:
        return path.charAt(0).toUpperCase() + path.slice(1);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col w-full overflow-x-hidden">
      {/* Sidebar Navigation */}
      <Sidebar
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed(!collapsed)}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 w-full">
        {/* Sticky Header */}
        <Header
          collapsed={collapsed}
          onToggleMobileMenu={() => setMobileOpen(true)}
        />

        {/* Content Container with dynamic left margin for sidebar */}
        <main
          className={`flex-1 transition-all duration-300 pb-12 min-w-0 ${
            collapsed ? 'lg:pl-20' : 'lg:pl-64'
          }`}
        >
          {/* Breadcrumb Navigation Bar */}
          <div className="bg-white/60 border-b border-slate-200/60 px-3 sm:px-6 lg:px-8 py-2 sm:py-2.5 flex items-center justify-between text-xs">
            <nav className="flex items-center gap-1.5 text-slate-500 overflow-x-auto whitespace-nowrap scrollbar-none py-0.5">
              <Link
                to="/dashboard"
                className="flex items-center gap-1 text-slate-500 hover:text-blue-600 transition-colors"
              >
                <Home className="w-3.5 h-3.5" />
                <span>Home</span>
              </Link>
              {pathnames.map((value, index) => {
                const to = `/${pathnames.slice(0, index + 1).join('/')}`;
                const isLast = index === pathnames.length - 1;
                return (
                  <React.Fragment key={to}>
                    <ChevronRight className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                    {isLast ? (
                      <span className="font-semibold text-slate-900">
                        {getBreadcrumbTitle(value)}
                      </span>
                    ) : (
                      <Link to={to} className="hover:text-blue-600 transition-colors">
                        {getBreadcrumbTitle(value)}
                      </Link>
                    )}
                  </React.Fragment>
                );
              })}
            </nav>

            <div className="hidden md:flex items-center gap-1.5 text-[11px] text-amber-700 font-medium bg-amber-50 px-2.5 py-0.5 rounded-full border border-amber-200/60 shrink-0 ml-2">
              <ShieldAlert className="w-3 h-3 text-amber-600 shrink-0" />
              <span>Research Environment &bull; Experimental Models</span>
            </div>
          </div>

          {/* Page View Slot */}
          <div className="px-3 sm:px-6 lg:px-8 py-4 sm:py-6 max-w-7xl mx-auto w-full">
            <Outlet />
          </div>
        </main>

        {/* Bottom Research Footer matching screenshot */}
        <footer
          className={`py-3.5 px-4 sm:px-6 lg:px-8 border-t border-slate-200/80 bg-white text-xs text-slate-500 transition-all duration-300 ${
            collapsed ? 'lg:pl-20' : 'lg:pl-64'
          }`}
        >
          <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-center sm:text-left">
            <p className="text-[11px] font-medium text-slate-500">
              Medical Image Classification <span className="mx-1 text-slate-300">|</span> ResNet50 vs EfficientNetB0 <span className="mx-1 text-slate-300">|</span> Transfer Learning
            </p>
            <p className="text-[11px] font-mono text-slate-400">
              v1.0.0
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
};
