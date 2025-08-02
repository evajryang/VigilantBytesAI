import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { ApiService } from '../services/api';
import { AnalyticsStats } from '../types';
import { useWebSocket } from '../services/WebSocketContext';

const DashboardContainer = styled.div`
  padding: 2rem;
  background: ${props => props.theme.colors.background};
  min-height: 100vh;
`;

const Header = styled.div`
  margin-bottom: 2rem;
`;

const Title = styled.h1`
  color: ${props => props.theme.colors.text.primary};
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
`;

const Subtitle = styled.p`
  color: ${props => props.theme.colors.text.secondary};
  font-size: 1.1rem;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const StatCard = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius};
  padding: 1.5rem;
  box-shadow: ${props => props.theme.shadows.sm};
  border: 1px solid ${props => props.theme.colors.border};
`;

const StatValue = styled.div`
  font-size: 2rem;
  font-weight: 700;
  color: ${props => props.theme.colors.text.primary};
  margin-bottom: 0.5rem;
`;

const StatLabel = styled.div`
  font-size: 0.9rem;
  color: ${props => props.theme.colors.text.secondary};
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const ConnectionStatus = styled.div<{ connected: boolean }>`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
  padding: 0.75rem 1rem;
  background: ${props => props.connected ? '#f0f9f4' : '#fef2f2'};
  border: 1px solid ${props => props.connected ? '#d1fae5' : '#fecaca'};
  border-radius: ${props => props.theme.borderRadius};
  color: ${props => props.connected ? '#065f46' : '#991b1b'};
`;

const StatusIndicator = styled.div<{ connected: boolean }>`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${props => props.connected ? '#10b981' : '#ef4444'};
  animation: ${props => props.connected ? 'pulse 2s infinite' : 'none'};
`;

const LoadingSpinner = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
  
  &::after {
    content: '';
    width: 40px;
    height: 40px;
    border: 4px solid ${props => props.theme.colors.border};
    border-top: 4px solid ${props => props.theme.colors.primary};
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const ErrorMessage = styled.div`
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #991b1b;
  padding: 1rem;
  border-radius: ${props => props.theme.borderRadius};
  margin-bottom: 1rem;
`;

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<AnalyticsStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { isConnected, connectionStatus } = useWebSocket();

  useEffect(() => {
    loadStats();
    const interval = setInterval(loadStats, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const loadStats = async () => {
    try {
      setError(null);
      const data = await ApiService.getAnalyticsStats(7);
      setStats(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load statistics');
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  if (loading) {
    return (
      <DashboardContainer>
        <LoadingSpinner />
      </DashboardContainer>
    );
  }

  return (
    <DashboardContainer>
      <Header>
        <Title>🛡️ VigilantBytes AI</Title>
        <Subtitle>Real-time fraud detection dashboard</Subtitle>
      </Header>

      <ConnectionStatus connected={isConnected}>
        <StatusIndicator connected={isConnected} />
        <span>
          Real-time monitoring: {isConnected ? 'Connected' : `Disconnected (${connectionStatus})`}
        </span>
      </ConnectionStatus>

      {error && (
        <ErrorMessage>
          ⚠️ {error}
        </ErrorMessage>
      )}

      {stats && (
        <StatsGrid>
          <StatCard>
            <StatValue>{stats.total_transactions.toLocaleString()}</StatValue>
            <StatLabel>Total Transactions (7d)</StatLabel>
          </StatCard>
          
          <StatCard>
            <StatValue>{stats.fraud_transactions.toLocaleString()}</StatValue>
            <StatLabel>Fraud Detected</StatLabel>
          </StatCard>
          
          <StatCard>
            <StatValue>{formatPercentage(stats.fraud_rate)}</StatValue>
            <StatLabel>Fraud Rate</StatLabel>
          </StatCard>
          
          <StatCard>
            <StatValue>{formatCurrency(stats.total_volume)}</StatValue>
            <StatLabel>Total Volume</StatLabel>
          </StatCard>
          
          <StatCard>
            <StatValue>{formatCurrency(stats.fraud_volume)}</StatValue>
            <StatLabel>Fraud Volume</StatLabel>
          </StatCard>
          
          <StatCard>
            <StatValue>{formatCurrency(stats.avg_transaction_amount)}</StatValue>
            <StatLabel>Avg Transaction</StatLabel>
          </StatCard>
        </StatsGrid>
      )}

      <div style={{ 
        background: 'white', 
        padding: '2rem', 
        borderRadius: '0.5rem', 
        textAlign: 'center',
        border: '1px solid #e2e8f0'
      }}>
        <h3 style={{ marginBottom: '1rem', color: '#1e293b' }}>
          🚀 VigilantBytes AI is Running!
        </h3>
        <p style={{ color: '#64748b', marginBottom: '1rem' }}>
          Your fraud detection system is actively monitoring transactions in real-time.
        </p>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
          gap: '1rem',
          marginTop: '1.5rem'
        }}>
          <div style={{ padding: '1rem', background: '#f8fafc', borderRadius: '0.5rem' }}>
            <strong>🔍 Transaction Monitor</strong>
            <br />
            <small>View and analyze all transactions</small>
          </div>
          <div style={{ padding: '1rem', background: '#f8fafc', borderRadius: '0.5rem' }}>
            <strong>🚨 Alert Center</strong>
            <br />
            <small>Manage fraud alerts and investigations</small>
          </div>
          <div style={{ padding: '1rem', background: '#f8fafc', borderRadius: '0.5rem' }}>
            <strong>📊 Analytics</strong>
            <br />
            <small>Deep dive into fraud patterns</small>
          </div>
          <div style={{ padding: '1rem', background: '#f8fafc', borderRadius: '0.5rem' }}>
            <strong>🧪 Test Transactions</strong>
            <br />
            <small>Submit test transactions for analysis</small>
          </div>
        </div>
      </div>
    </DashboardContainer>
  );
};

export default Dashboard;