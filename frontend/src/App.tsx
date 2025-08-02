import React, { useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import styled, { ThemeProvider, createGlobalStyle } from 'styled-components';

import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import TransactionMonitor from './components/TransactionMonitor';
import AlertCenter from './components/AlertCenter';
import Analytics from './components/Analytics';
import ModelManagement from './components/ModelManagement';
import TransactionTester from './components/TransactionTester';
import { theme } from './utils/theme';

const GlobalStyle = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
      'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    background: ${props => props.theme.colors.background};
    color: ${props => props.theme.colors.text.primary};
    line-height: 1.6;
  }

  code {
    font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New', monospace;
  }

  /* Scrollbar styling */
  ::-webkit-scrollbar {
    width: 6px;
    height: 6px;
  }

  ::-webkit-scrollbar-track {
    background: #f1f5f9;
  }

  ::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }

  /* Custom animations */
  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes slideIn {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(0);
    }
  }

  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.7;
    }
  }

  .fade-in {
    animation: fadeIn 0.3s ease-out;
  }

  .slide-in {
    animation: slideIn 0.3s ease-out;
  }

  .pulse {
    animation: pulse 2s infinite;
  }

  /* Utility classes */
  .text-center {
    text-align: center;
  }

  .flex {
    display: flex;
  }

  .flex-column {
    flex-direction: column;
  }

  .justify-center {
    justify-content: center;
  }

  .align-center {
    align-items: center;
  }

  .space-between {
    justify-content: space-between;
  }

  .full-width {
    width: 100%;
  }

  .full-height {
    height: 100%;
  }

  .relative {
    position: relative;
  }

  .absolute {
    position: absolute;
  }

  .fixed {
    position: fixed;
  }

  .overflow-hidden {
    overflow: hidden;
  }

  .overflow-auto {
    overflow: auto;
  }

  .cursor-pointer {
    cursor: pointer;
  }

  .no-select {
    user-select: none;
  }

  /* Responsive breakpoints */
  @media (max-width: 768px) {
    .mobile-hidden {
      display: none !important;
    }
  }

  @media (min-width: 769px) {
    .desktop-hidden {
      display: none !important;
    }
  }
`;

const AppContainer = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
`;

const App: React.FC = () => {
  useEffect(() => {
    // Hide loading screen
    const loadingElement = document.getElementById('loading');
    if (loadingElement) {
      setTimeout(() => {
        loadingElement.style.opacity = '0';
        loadingElement.style.transition = 'opacity 0.5s ease-out';
        setTimeout(() => {
          loadingElement.style.display = 'none';
        }, 500);
      }, 1500);
    }
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <AppContainer>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="transactions" element={<TransactionMonitor />} />
            <Route path="alerts" element={<AlertCenter />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="models" element={<ModelManagement />} />
            <Route path="test" element={<TransactionTester />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Route>
        </Routes>
      </AppContainer>
    </ThemeProvider>
  );
};

export default App;