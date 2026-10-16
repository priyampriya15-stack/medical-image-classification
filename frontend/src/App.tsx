import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './auth/AuthContext';
import { ProtectedRoute } from './auth/ProtectedRoute';
import { AppLayout } from './components/layout/AppLayout';

// Pages
import { LoginPage } from './pages/LoginPage';
import { DashboardPage } from './pages/DashboardPage';
import { DatasetExplorerPage } from './pages/DatasetExplorerPage';
import { ModelTrainingPage } from './pages/ModelTrainingPage';
import { ModelComparisonPage } from './pages/ModelComparisonPage';
import { ImageClassifierPage } from './pages/ImageClassifierPage';
import { GradCamPage } from './pages/GradCamPage';
import { TrainingAnalyticsPage } from './pages/TrainingAnalyticsPage';
import { ProjectInfoPage } from './pages/ProjectInfoPage';

import { useAuth } from './auth/AuthContext';

const RootRedirect: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuth();
  if (isLoading) return null;
  return <Navigate to={isAuthenticated ? "/dashboard" : "/login"} replace />;
};

export const App: React.FC = () => {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Login Route */}
          <Route path="/login" element={<LoginPage />} />

          {/* Protected Application Routes */}
          <Route element={<ProtectedRoute />}>
            <Route element={<AppLayout />}>
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/dataset" element={<DatasetExplorerPage />} />
              <Route path="/training" element={<ModelTrainingPage />} />
              <Route path="/comparison" element={<ModelComparisonPage />} />
              <Route path="/model-comparison" element={<ModelComparisonPage />} />
              <Route path="/classifier" element={<ImageClassifierPage />} />
              <Route path="/gradcam" element={<GradCamPage />} />
              <Route path="/analytics" element={<TrainingAnalyticsPage />} />
              <Route path="/training-analytics" element={<TrainingAnalyticsPage />} />
              <Route path="/project-info" element={<ProjectInfoPage />} />
            </Route>
          </Route>

          {/* Default Redirections */}
          <Route path="/" element={<RootRedirect />} />
          <Route path="*" element={<RootRedirect />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
};

export default App;
