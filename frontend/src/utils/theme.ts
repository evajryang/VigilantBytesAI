import { Theme } from '../types';

export const theme: Theme = {
  colors: {
    primary: '#667eea',
    secondary: '#764ba2',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#3b82f6',
    background: '#f8fafc',
    surface: '#ffffff',
    text: {
      primary: '#1e293b',
      secondary: '#64748b',
      disabled: '#94a3b8',
    },
    border: '#e2e8f0',
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
  },
  borderRadius: '0.5rem',
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  },
};

// Helper functions for theme usage
export const getSeverityColor = (severity: string) => {
  switch (severity) {
    case 'low':
      return theme.colors.success;
    case 'medium':
      return theme.colors.warning;
    case 'high':
      return '#ff6b35';
    case 'critical':
      return theme.colors.error;
    default:
      return theme.colors.text.secondary;
  }
};

export const getStatusColor = (status: string) => {
  switch (status) {
    case 'active':
      return theme.colors.error;
    case 'investigated':
      return theme.colors.warning;
    case 'resolved':
      return theme.colors.success;
    case 'false_positive':
      return theme.colors.text.secondary;
    default:
      return theme.colors.text.secondary;
  }
};